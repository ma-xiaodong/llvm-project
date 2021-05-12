#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include "llvm/Support/Debug.h"

#include <sstream>
#include <unordered_map>
#include <iostream>

#define DEBUG_TYPE    "hopt"
#define CACHE_UNIT    1024

using namespace mlir;
using namespace memref;

namespace {
struct HigherOrderPolyhedralOpt
    : public HigherOrderPolyhedralOptBase<HigherOrderPolyhedralOpt> {

  void runOnFunction() override;
  void runOnBlock(Block *block);
};
}

static unsigned getMatmulOptParameter(Operation *op, StringRef name) {
  IntegerAttr attr = op->getAttrOfType<IntegerAttr>(name);

  assert(attr && "optimization parameter not found");
  return attr.getValue().getSExtValue();
}

#define vecBytes    (256/8)
// Compute the tile size from the matrix and architecture params:
// 1. M: height of matrix A;          2. N: width of matrix B;
// 3. K: width of A, height of B;     4. L1S: L1 cache size;
// 5. L2S: L2 cache size;             6. L3S: L3 cache size
// 7. RS: register number of the cpu'
// We use byteWidth to represent the element size of the matrix.
// vecBytes to represent vector byte width.
// All the above numbers are times of K.
//
// The variables need to be computed:
// kc : tile size along K dimension
// mc : tile size along M dimension
// nr : tile size along N dimension
// mr : tile size within mc
//
// The constraints they must satisfied:
// 0 <= (mc * kc + kc * N + mc * N) * byteWidth <= L3S          (1)
// 0 <= (mc * kc + kc * nr + mc * nr)* byteWidth <= L2S         (2)
// 0 <= ((mr + nr) * kc + mr * nr) * byteWidth <= L1S           (3)
// mr + 1 + mr * nr / (vecBytes / byteWidth) = RS               (4)
// 0 <= kc <= K                                                 (5)
// 0 <= mc <= M                                                 (6)
// 0 <= mr <= mc                                                (7)
// 0 <= nr <= N                                                 (8)
// K % kc = 0                                                   (9)
// N % nr = 0                                                   (10)
// M % mc = 0                                                   (11)
// mc % mr = 0                                                  (12)

static SmallVector<unsigned, 4> computeTileSize(AffineForOp *forOp) {
  // The order of computed size: kc, 
  SmallVector<unsigned, 4> computedSize;
  unsigned M, N, K, RS, L1S, L2S, L3S, byteWidth;
  
  M = getMatmulOptParameter(*forOp, "M");
  N = getMatmulOptParameter(*forOp, "N");
  K = getMatmulOptParameter(*forOp, "K");
  L1S = getMatmulOptParameter(*forOp, "L1S");
  L2S = getMatmulOptParameter(*forOp, "L2S");
  L3S = getMatmulOptParameter(*forOp, "L3S");
  RS = getMatmulOptParameter(*forOp, "RS");

  // Get the element size of the matrix
  forOp->walk([&](AffineLoadOp loadOp) {
    auto memrefType = loadOp.memref().getType().cast<MemRefType>();
    byteWidth = memrefType.getElementType().getIntOrFloatBitWidth() / 8;
  });

  unsigned vecSize = vecBytes / byteWidth;
  //unsigned vecSize = 1;
  unsigned kc, mc, mr, nr;

  // Compute mr and nr. First find all the pairs (mr, nr) which satifies (4) 
  // assign mr suitable from 2
  std::vector<std::pair<unsigned, unsigned>> rTmpStack, rFinalStack;
  mr = 2;
  nr = (RS - 1 - mr) * vecSize / mr;
  // add all possible (mr, nr) to rTmpStack
  while (nr >= vecSize) {
    rTmpStack.push_back(std::make_pair(mr, nr));
    mr++;
    nr = (RS - 1 - mr) * vecSize / mr;
  }

  // choose mr that satisfies M % mr == 0
  for (unsigned i = 0; i < rTmpStack.size(); i++) {
    mr = rTmpStack[i].first;
    nr = rTmpStack[i].second;
    if (M % mr == 0 && nr % vecSize == 0) {
      rFinalStack.push_back(std::make_pair(mr, nr));
    }
  }
  // if M % mr != 0, reassign mr
  if (rFinalStack.size() == 0) {
    mr = 2;
    nr = (RS - 1 - mr) * vecSize / mr;
  } else {
    // use (4) to choose the pair whose result is colsest to RS
    mr = rFinalStack[0].first;
    nr = rFinalStack[0].second;
    int diff = RS - (mr + 1 + mr * nr / vecSize);

    for (unsigned i = 1; i < rFinalStack.size(); i++) {
      int tmpMr = rFinalStack[i].first;
      int tmpNr = rFinalStack[i].second;
      int tmpDiff = RS - (tmpMr + 1 + tmpMr * tmpNr / vecSize);

      if (diff > tmpDiff) {
        mr = tmpMr;
        nr = tmpNr;
        diff = tmpDiff;
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "final (mr, nr): " << "(" << mr << ", " << nr << ")\n");
  // check nr satisfies contition (8)
  assert(nr <= N && "N is too small");

  // Compute kc. First use (3) to get the max value, then check if it satifies
  // (1). Then check constraints (5).
  kc = (L1S * CACHE_UNIT / byteWidth - mr * nr) / (mr + nr);
  assert(kc <= K && "K is too small");

  // Compute mc using constraint (2)
  // mc = L2S * CACHE_UNIT / (kc * byteWidth);
  mc = (L2S * CACHE_UNIT / byteWidth - kc * nr) / (nr + kc);
  if (mc < mr) {
    // this condition looks impossible
    mc = mr;
  }
  assert(mc <= M && "M is too small");

  // Check whether L3 is enough
  if ((mc * kc + kc * N + mc * N) * byteWidth > L3S * CACHE_UNIT) {
    LLVM_DEBUG(llvm::dbgs() << "Warining L3 Cache is small?\n");
  }

  return computedSize = {kc, mc, mr, nr};
}

void HigherOrderPolyhedralOpt::runOnFunction() {
  auto func = getFunction();

  for (auto &block : func) {
    runOnBlock(&block);
  }
  {
    auto *context = &getContext();
    OwningRewritePatternList patterns(context);
    AffineLoadOp::getCanonicalizationPatterns(patterns, context);
    AffineStoreOp::getCanonicalizationPatterns(patterns, context);
    AffineApplyOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      llvm::errs() << "error in applyPatternsAndFoldGreedily\n";
    }
  }
}

void HigherOrderPolyhedralOpt::runOnBlock(Block *block) {
  StringAttr polyClass;
  SmallVector<AffineForOp, 7> band;
  // find the memref of matrix A, B, C
  Value outputMemRef, lhsMemRef, rhsMemRef;
  // the tile size
  unsigned kc, mc, mr, nr;

  for (auto &op : *block) {
    if (auto forOp = dyn_cast<AffineForOp>(op)) {
      polyClass = forOp.getOperation()->getAttrOfType<StringAttr>("class");

      if (!polyClass) {
        continue;
      }
      if (polyClass.getValue().equals("matmul")) {
        getPerfectlyNestedLoops(band, forOp);
        break;
      }
    }
  }
  if (!polyClass) {
    // Unrelated loop
    return;
  }
  assert(band.size() == 3 && "matmul has at most 3 loops");

  OpBuilder builder(band[0]);

  band[2].walk([&](AffineStoreOp storeOp){
    outputMemRef = storeOp.getMemRef();
  });
  band[2].walk([&](AffineLoadOp loadOp){
    if (outputMemRef == loadOp.getMemRef()) {
      return;
    }
    rhsMemRef = loadOp.getMemRef();
  });
  band[2].walk([&](AffineLoadOp loadOp){
    if (loadOp.getMemRef() == outputMemRef || loadOp.getMemRef() == rhsMemRef) {
      return;
    }
    lhsMemRef = loadOp.getMemRef();
  });


  // Tiling the loops in matrix multiplication
  if (clTile) {
    // Original loop
    // Result of first tiling
    SmallVector<AffineForOp, 7> tiledNest0;
    // Result of second tiling
    SmallVector<AffineForOp, 7> tiledNest1;

    SmallVector<unsigned, 4> tileSize = computeTileSize(&band[0]);
    LLVM_DEBUG(llvm::dbgs() << "kc, mc, mr, nr: "
                            << tileSize[0] << ", " << tileSize[1] << ", " 
                            << tileSize[2] << ", " << tileSize[3] << "\n");

    kc = tileSize[0]; mc = tileSize[1];
    mr = tileSize[2]; nr = tileSize[3];

    // Dimension should be tiled twice.
    if (failed(tilePerfectlyNested(band, {mr, nr, kc}, &tiledNest1))) {
      LLVM_DEBUG(llvm::dbgs() << "failed during first tiling");
    }
    // Tile mc further
    if (failed(tilePerfectlyNested(tiledNest1[0], 
                                   mc / mr, &tiledNest0))) {
      LLVM_DEBUG(llvm::dbgs() << "failed during second tiling");
    }

    // After tiling, interchange the loops. 
    // The original order of loop vars are:
    // ioo, ioi, jo, ko, ii, ji, ki
    // The inchanged order of loop vars should be:
    // ko, ioo, jo, ioi, ki, ji, ii

    // Interchange ji with ki
    tiledNest1.clear();
    getPerfectlyNestedLoops(tiledNest1, tiledNest0[0]);
    interchangeLoops(tiledNest1[5], tiledNest1[6]);

    // Interchange ii with ki
    tiledNest1.clear();
    getPerfectlyNestedLoops(tiledNest1, tiledNest0[0]);
    interchangeLoops(tiledNest1[4], tiledNest1[5]);

    // Interchange ii with ji
    tiledNest1.clear();
    getPerfectlyNestedLoops(tiledNest1, tiledNest0[0]);
    interchangeLoops(tiledNest1[5], tiledNest1[6]);

    // Interchange ioi with jo
    tiledNest1.clear();
    getPerfectlyNestedLoops(tiledNest1, tiledNest0[0]);
    interchangeLoops(tiledNest1[1], tiledNest1[2]);

    // Interchange ioi with ko
    tiledNest1.clear();
    getPerfectlyNestedLoops(tiledNest1, tiledNest0[0]);
    interchangeLoops(tiledNest1[2], tiledNest1[3]);

    // Interchange jo with ko
    tiledNest1.clear();
    getPerfectlyNestedLoops(tiledNest1, tiledNest0[0]);
    interchangeLoops(tiledNest1[1], tiledNest1[2]);

    // Interchange ioo with ko
    tiledNest1.clear();
    getPerfectlyNestedLoops(tiledNest1, tiledNest0[0]);
    interchangeLoops(tiledNest1[0], tiledNest1[1]);

    // resume the loop to band
    band.clear();
    getPerfectlyNestedLoops(band, tiledNest1[1]);
    assert(band.size() == 7 && "number of tiled loops are incorrect.");
  }

  if (clUnroll) {
    if (band.size() != 7) {
      LLVM_DEBUG(llvm::dbgs() << "unroll must follows tiling"); 
    }
    if (failed(loopUnrollJamUpToFactor(band[6], mr))) {
      llvm::errs() << "failed in unrolling mr\n";
      return;
    }
    if (failed(loopUnrollJamUpToFactor(band[5], nr))) {
      llvm::errs() << "failed in unrolling nr\n";
      return;
    }
    // does not unroll the ki loop again, which is unrolled in uday's paper.
  }

  if (clCopy) {
    unsigned byteWidth = lhsMemRef.getType().cast<MemRefType>().
                                   getElementType().getIntOrFloatBitWidth() / 8;
    AffineCopyOptions copyOptions = {false,
                                     0,
                                     0,
                                     0,
                                     2 * 1024 * 1024,
                                     };
    DenseSet<Operation *> copyNests;
    /// band[1] is the loop of ioo, allocate and copy memory of size mc * kc of 
    /// matrix A in L2.
    copyOptions.fastMemCapacityBytes = mc * kc * byteWidth; 
    affineDataCopyGenerate(band[1].getBody()->begin(),
                           std::prev(band[1].getBody()->end()),
                           copyOptions,
                           lhsMemRef,
                           copyNests);

    // band[0] is the loop of ko, allocate and copy memory of size kc * N of 
    // matrix B in L3.
    if (getConstantTripCount(band[0]).hasValue()) {
      unsigned N = getConstantTripCount(band[0]).getValue();
      copyOptions.fastMemCapacityBytes = kc * N * band[0].getStep() * byteWidth;
    } else {
      llvm::errs() << "error in the k loop of matrix B\n";
      return;
    }
    copyNests.clear();
    affineDataCopyGenerate(band[0].getBody()->begin(),
                           std::prev(band[0].getBody()->end()), 
                           copyOptions,
                           rhsMemRef,
                           copyNests);

    // band[2] is the loop of ji, allocate and copy memory of size nr * kc of
    // matrix B in L1, which is a subblock of matrix B in L3, so we need to find
    // it.
    Value rhsL3Buf;
    // the copy loop generated by the above affineDataCopyGenerate functio 
    // should be a AffineForOp
    auto forOp = dyn_cast<AffineForOp>(*copyNests.begin());
    assert(forOp && "wrong result of affineDataCopyGenerate");
    forOp.walk([&](AffineStoreOp storeOp) {
      rhsL3Buf = storeOp.getMemRef();
    });

    copyNests.clear();
    copyOptions.fastMemCapacityBytes = kc * nr * byteWidth;
    affineDataCopyGenerate(band[2].getBody()->begin(),
                           std::prev(band[2].getBody()->end()),
                           copyOptions,
                           rhsL3Buf,
                           copyNests);

    // TODO: Set alignment to 256-bit boundaries
  }

  if (clScalRep) {
    block->walk([&](AffineForOp forOp) { scalarReplace(forOp); });
  }
  return;
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createHigherOrderPolyhedralOptPass() {
  return std::make_unique<HigherOrderPolyhedralOpt>();
}
