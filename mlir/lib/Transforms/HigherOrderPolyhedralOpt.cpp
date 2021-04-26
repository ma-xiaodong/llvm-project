#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/Utils.h"
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

#define DEBUG_TYPE "hopt"

using namespace mlir;

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
// 0 <= kc * N * byteWidth <= L3S                   (1)
// 0 <= mc * kc * byteWidth <= L2S                  (2)
// 0 <= (mr + nr) * kc * byteWidth <= L1S           (3)
// mr + 1 + mr * nr / (vecBytes / byteWidth) = RS   (4)
// 0 <= kc <= K                                     (5)
// 0 <= mc <= M                                     (6)
// 0 <= mr <= mc                                    (7)
// 0 <= nr <= N                                     (8)
// K % kc = 0                                       (9)
// N % nr = 0                                       (10)
// M % mc = 0                                       (11)
// mc % mr = 0                                      (12)

static SmallVector<unsigned, 4> computerTileSize(AffineForOp *forOp) {
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
  unsigned kc, mc, mr, nr;

  // Compute mr and nr. First find all the pairs (mr, nr) which satifies (4) 
  std::vector<std::pair<unsigned, unsigned>> rTileStack;
  nr = vecSize;
  mr = (RS - 1) / (1 + nr / vecSize);
  rTileStack.push_back(std::make_pair(mr, nr));
  {
    unsigned oldDiff, newDiff;
    oldDiff = RS - (mr + 1 + mr * nr / vecSize);

    while(mr > 0 && nr <= N) {
      nr += vecSize;
      mr = (RS - 1) / (1 + nr / vecSize);
      newDiff = RS - (mr + 1 + mr * nr / vecSize);

      if (newDiff < oldDiff) {
        while(rTileStack.size()) {
          rTileStack.pop_back();
        }
        rTileStack.push_back(std::make_pair(mr, nr));
        oldDiff = newDiff;
      } else if (newDiff == oldDiff) {
        rTileStack.push_back(std::make_pair(mr, nr));
      }
    }
  }

  // There maybe more then one element in rTileStack, choose the one which
  // minimize |mr-nr|
  if (rTileStack.size() > 1) {
    unsigned oldDiff, newDiff, tmpMr, tmpNr;

    mr = rTileStack[rTileStack.size() - 1].first;
    nr = rTileStack[rTileStack.size() - 1].second;
    oldDiff = (mr > nr)? (mr -nr) : (nr - mr);
    rTileStack.pop_back();
    LLVM_DEBUG(llvm::dbgs() << "(mr, nr): " << "(" << mr << ", " << nr << ")\n");

    while (rTileStack.size()) {
      tmpMr = rTileStack[rTileStack.size() - 1].first;
      tmpNr = rTileStack[rTileStack.size() - 1].second;
      LLVM_DEBUG(llvm::dbgs() << "(mr, nr): " << "(" << tmpMr << ", " << tmpNr << ")\n");

      newDiff = (tmpMr > tmpNr) ? (tmpMr - tmpNr) : (tmpNr - tmpMr);

      if (newDiff < oldDiff) {
        mr = tmpMr;
	nr = tmpNr;
	oldDiff = newDiff;
      }
      rTileStack.pop_back();
    }
  } else {
    mr = rTileStack[0].first;
    nr = rTileStack[0].second;
  }
  LLVM_DEBUG(llvm::dbgs() << "final (mr, nr): " << "(" << mr << ", " << nr << ")\n");

  return computedSize;
}

void HigherOrderPolyhedralOpt::runOnFunction() {
  auto func = getFunction();

  for (auto &block : func) {
    runOnBlock(&block);
  }
}

void HigherOrderPolyhedralOpt::runOnBlock(Block *block) {
  SmallVector<AffineForOp, 3> band;
  StringAttr polyClass;

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
  if (clTile) {
    SmallVector<unsigned, 4> tileSize = computerTileSize(&band[0]);
  }
  return;
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createHigherOrderPolyhedralOptPass() {
  return std::make_unique<HigherOrderPolyhedralOpt>();
}
