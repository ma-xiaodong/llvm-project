func @main() {
  %A = memref.alloc() : memref<2088x2048xf64>
  %B = memref.alloc() {alignment = 32} : memref<2048x2048xf64>
  %C = memref.alloc() {alignment = 32} : memref<2088x2048xf64>

  %cf1 = constant 1.000000e+00 : f64
  linalg.fill(%A, %cf1) : memref<2088x2048xf64>, f64
  linalg.fill(%B, %cf1) : memref<2048x2048xf64>, f64

  %reps = constant 5 : index
  affine.for %ti = 0 to %reps {
    linalg.fill(%C, %cf1) : memref<2088x2048xf64>, f64
    call @matmul(%A, %B, %C) : (memref<2088x2048xf64>, memref<2048x2048xf64>, memref<2088x2048xf64>) -> ()
  }

  %pC = memref.cast %C : memref<2088x2048xf64> to memref<*xf64>
  call @print_memref_f64(%pC) : (memref<*xf64>) -> ()

  return
}

func @matmul(%arg0 : memref<2088x2048xf64>, %arg1 : memref<2048x2048xf64>, %arg2 : memref<2088x2048xf64>) {
  affine.for %i = 0 to 2088 {
    affine.for %j = 0 to 2048 {
      affine.for %k = 0 to 2048 {
        %0 = affine.load %arg0[%i, %k] : memref<2088x2048xf64>
        %1 = affine.load %arg1[%k, %j] : memref<2048x2048xf64>
        %2 = affine.load %arg2[%i, %j] : memref<2088x2048xf64>
	%3 = mulf %0, %1 : f64
	%4 = addf %3, %2 : f64
	affine.store %4, %arg2[%i, %j] : memref<2088x2048xf64>
      }
    }
  } {class = "matmul", M = 2088, N = 2048, K = 2048, L1S = 32, L2S = 256, L3S = 12288, RS=16}
  return
}
func private @print_memref_f64(memref<*xf64>)
