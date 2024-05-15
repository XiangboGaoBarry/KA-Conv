# mkdir logs
mkdir logs

# CUDA_VISIBLE_DEVICES=0 python3 train_cifar10.py --model kanconv --kan_type RBF > logs/kanconv_RBF.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 python3 train_cifar10.py --model kanconv_small --kan_type RBF > logs/kanconv_small_RBF.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 python3 train_cifar10.py --model mlp > logs/mlp.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar10.py --model kanconv --kan_type Poly > logs/kanconv_Poly.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python3 train_cifar10.py --model kanconv_small --kan_type Poly > logs/kanconv_small_Poly.log 2>&1  &
# CUDA_VISIBLE_DEVICES=4 python3 train_cifar10.py --model kanconv --kan_type Chebyshev > logs/kanconv_Chebyshev.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 python3 train_cifar10.py --model kanconv_small --kan_type Chebyshev > logs/kanconv_small_Chebyshev.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 python3 train_cifar10.py --model kanconv --kan_type Fourier > logs/kanconv_Fourier.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 python3 train_cifar10.py --model kanconv_small --kan_type Fourier > logs/kanconv_small_Fourier.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 python3 train_cifar10.py --model kanconv --kan_type BSpline > logs/kanconv_BSpline.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 python3 train_cifar10.py --model kanconv_small --kan_type BSpline > logs/kanconv_small_BSpline.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 python3 train_cifar10.py --model convkan_efficient > logs/convkan_efficient.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 python3 train_cifar10.py --model convkan_fast > logs/convkan_fast.log 2>&1 &


# tiny
CUDA_VISIBLE_DEVICES=3 python3 train_cifar10.py --model kanconv_tiny --kan_type Poly > logs/kanconv_tiny_Poly.log 2>&1  &
CUDA_VISIBLE_DEVICES=4 python3 train_cifar10.py --model kanconv_tiny --kan_type Chebyshev > logs/kanconv_tiny_Chebyshev.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 python3 train_cifar10.py --model kanconv_tiny --kan_type Fourier > logs/kanconv_tiny_Fourier.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 python3 train_cifar10.py --model kanconv_tiny --kan_type BSpline > logs/kanconv_tiny_BSpline.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 train_cifar10.py --model kanconv_tiny --kan_type RBF > logs/kanconv_tiny_RBF.log 2>&1 &