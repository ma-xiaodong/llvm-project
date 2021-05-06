#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0 + 4)>
module  {
  func @main() {
    %0 = memref.alloc() : memref<2088x2048xf32>
    %1 = memref.alloc() {alignment = 32 : i64} : memref<2048x2048xf32>
    %2 = memref.alloc() {alignment = 32 : i64} : memref<2088x2048xf32>
    %cst = constant 1.000000e+00 : f32
    linalg.fill(%0, %cst) : memref<2088x2048xf32>, f32 
    linalg.fill(%1, %cst) : memref<2048x2048xf32>, f32 
    %c5 = constant 5 : index
    %3 = call @rtclock() : () -> f64
    affine.for %arg0 = 0 to %c5 {
      affine.for %arg1 = #map0(%arg0) to #map1(%arg0) {
        linalg.fill(%2, %cst) : memref<2088x2048xf32>, f32 
        call @matmul(%0, %1, %2) : (memref<2088x2048xf32>, memref<2048x2048xf32>, memref<2088x2048xf32>) -> ()
      }
    }
    %4 = call @rtclock() : () -> f64
    %5 = memref.cast %2 : memref<2088x2048xf32> to memref<*xf32>
    call @print_memref_f32(%5) : (memref<*xf32>) -> ()
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c2088 = constant 2088 : index
    %c2048 = constant 2048 : index
    %c2048_0 = constant 2048 : index
    %6 = subf %4, %3 : f64
    %7 = muli %c2088, %c2048 : index
    %8 = muli %7, %c2048_0 : index
    %c2 = constant 2 : index
    %9 = muli %c2, %8 : index
    %10 = muli %c5, %9 : index
    %11 = index_cast %10 : index to i64
    %12 = sitofp %11 : i64 to f64
    %13 = divf %12, %6 : f64
    call @print_flops(%13) : (f64) -> ()
    return
  }
  func @matmul(%arg0: memref<2088x2048xf32>, %arg1: memref<2048x2048xf32>, %arg2: memref<2088x2048xf32>) {
    affine.for %arg3 = 0 to 261 {
      affine.for %arg4 = 0 to 256 {
        affine.for %arg5 = 0 to 256 {
          affine.for %arg6 = 0 to 8 {
            affine.for %arg7 = 0 to 8 {
              affine.for %arg8 = 0 to 8 {
                %0 = affine.load %arg0[%arg3 * 8 + %arg6, %arg5 * 8 + %arg8] : memref<2088x2048xf32>
                %1 = affine.load %arg1[%arg5 * 8 + %arg8, %arg4 * 8 + %arg7] : memref<2048x2048xf32>
                %2 = affine.load %arg2[%arg3 * 8 + %arg6, %arg4 * 8 + %arg7] : memref<2088x2048xf32>
                %3 = mulf %0, %1 : f32
                %4 = addf %3, %2 : f32
                affine.store %4, %arg2[%arg3 * 8 + %arg6, %arg4 * 8 + %arg7] : memref<2088x2048xf32>
              }
            }
          }
        }
      }
    }
    return
  }
  func private @print_flops(f64)
  func private @rtclock() -> f64
  func private @print_memref_f32(memref<*xf32>)
}

