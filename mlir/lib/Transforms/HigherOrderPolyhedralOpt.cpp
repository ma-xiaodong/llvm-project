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

// Compute the tile size from the matrix and architecture params:
// 1. M: height of matrix A;          2. N: width of matrix B;
// 3. K: width of A, height of B;     4. L1S: L1 cache size;
// 5. L2S: L2 cache size;             6. L3S: L3 cache size
// 7. RS: register number of the cpu'
// All the above numbers are times of K
//
// The variables need to be computed:
// kc : tile size along K dimension
// mc : tile size along M dimension
// nr : tile size along N dimension
// mr : tile size within mc
//
// The constraints they must satisfied:
// 0 <= kc * N <= L3S          (1)
// 0 <= mc * kc <= L2S         (2)
// 0 <= (mr + nr) * kc <= L1S  (3)
// mr + nr = RS                (4)
// 0 <= kc <= K                (5)
// 0 <= mc <= M                (6)
// 0 <= mr <= mc               (7)
// 0 <= nr <= N                (8)
// K % kc = 0                  (9)
// N % nr = 0                  (10)
// M % mc = 0                  (11)
// mc % mr = 0                 (12)

static SmallVector<unsigned, 4> computerTileSize(AffineForOp *forOp) {
  // The order of computed size: kc, 
  SmallVector<unsigned, 4> computedSize;
  unsigned M, N, K, RS, L1S, L2S, L3S;
  
  M = getMatmulOptParameter(*forOp, "M");
  N = getMatmulOptParameter(*forOp, "N");
  K = getMatmulOptParameter(*forOp, "K");
  L1S = getMatmulOptParameter(*forOp, "L1S");
  L2S = getMatmulOptParameter(*forOp, "L2S");
  L3S = getMatmulOptParameter(*forOp, "L3S");
  RS = getMatmulOptParameter(*forOp, "RS");

  unsigned upper_kc1 = L1S / RS;
  LLVM_DEBUG(llvm::dbgs() << "First upper of KC: " << upper_kc1 << "\n");

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
