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

  unsigned l1CacheSize, l2CacheSize, l3CacheSize, M, N, K;

  l1CacheSize = getMatmulOptParameter(band[0], "L1_C");
  l2CacheSize = getMatmulOptParameter(band[0], "L2_C");
  l3CacheSize = getMatmulOptParameter(band[0], "L3_C");
  M = getMatmulOptParameter(band[0], "M");
  N = getMatmulOptParameter(band[0], "N");
  K = getMatmulOptParameter(band[0], "K");

  if (clTile) {
    // compute the tile param form machine param
    unsigned kc, mc, mr, nr;
  }
  return;
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createHigherOrderPolyhedralOptPass() {
  return std::make_unique<HigherOrderPolyhedralOpt>();
}
