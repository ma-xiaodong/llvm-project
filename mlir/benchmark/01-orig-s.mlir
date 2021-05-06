func @main() {
  %A = memref.alloc() : memref<2088x2048xf32>
  %B = memref.alloc() {alignment = 32} : memref<2048x2048xf32>
  %C = memref.alloc() {alignment = 32} : memref<2088x2048xf32>

  %cf1 = constant 1.000000e+00 : f32
  linalg.fill(%A, %cf1) : memref<2088x2048xf32>, f32
  linalg.fill(%B, %cf1) : memref<2048x2048xf32>, f32

  %reps = constant 5 : index
  %t_start = call @rtclock() : () -> (f64)
  affine.for %ti = 0 to %reps {
    linalg.fill(%C, %cf1) : memref<2088x2048xf32>, f32
    call @matmul(%A, %B, %C) : (memref<2088x2048xf32>, memref<2048x2048xf32>, memref<2088x2048xf32>) -> ()
  }
  %t_end = call @rtclock() : () -> (f64)

  %pC = memref.cast %C : memref<2088x2048xf32> to memref<*xf32>
  call @print_memref_f32(%pC) : (memref<*xf32>) -> ()

  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %M = constant 2088 : index
  %N = constant 2048 : index
  %K = constant 2048 : index

  %t = subf %t_end, %t_start : f64
  %f1 = muli %M, %N : index
  %f2 = muli %f1, %K : index
  // 2*M*N*K
  %c2 = constant 2 : index
  %f3 = muli %c2, %f2 : index
  %num_flops = muli %reps, %f3 : index
  %num_flops_i = index_cast %num_flops : index to i64
  %num_flops_f = sitofp %num_flops_i : i64 to f64
  %flops = divf %num_flops_f, %t : f64
  call @print_flops(%flops) : (f64) -> ()

  return
}

func @matmul(%arg0 : memref<2088x2048xf32>, %arg1 : memref<2048x2048xf32>, %arg2 : memref<2088x2048xf32>) {
  affine.for %i = 0 to 2088 {
    affine.for %j = 0 to 2048 {
      affine.for %k = 0 to 2048 {
        %0 = affine.load %arg0[%i, %k] : memref<2088x2048xf32>
        %1 = affine.load %arg1[%k, %j] : memref<2048x2048xf32>
        %2 = affine.load %arg2[%i, %j] : memref<2088x2048xf32>
  	    %3 = mulf %0, %1 : f32
  	    %4 = addf %3, %2 : f32
	    affine.store %4, %arg2[%i, %j] : memref<2088x2048xf32>
      }
    }
  } {class = "matmul", M = 2088, N = 2048, K = 2048, L1S = 32, L2S = 256, L3S = 12288, RS=16}
  return
}

func private @print_flops(f64)
func private @rtclock() -> f64
func private @print_memref_f32(memref<*xf32>)
