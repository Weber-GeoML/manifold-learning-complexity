#####################################################
# SYNTHETIC DATASETS
#####################################################

python train_diffusion_model_synthetic.py --intrinsic_dim 2 --ambient_dim 20 --num_points 100000 --num_epochs 30 --batch_size 64 --lr 4e-4 --save_path models/sphere_2_20
python generate_from_diffusion_synthetic.py --model models/sphere_2_20 --intrinsic_dim 2 --ambient_dim 20 --num_samples 100000 --batch_size 256 --save_path "manifolds/sphere_2_20_t=20" --timestep 20
python train_diffusion_model_synthetic.py --intrinsic_dim 10 --ambient_dim 20 --num_points 100000 --num_epochs 30 --batch_size 64 --lr 4e-4 --save_path models/sphere_10_20
python generate_from_diffusion_synthetic.py --model models/sphere_10_20 --intrinsic_dim 10 --ambient_dim 20 --num_samples 100000 --batch_size 256 --save_path "manifolds/sphere_10_20_t=20" --timestep 20
python train_diffusion_model_synthetic.py --intrinsic_dim 18 --ambient_dim 20 --num_points 100000 --num_epochs 30 --batch_size 64 --lr 4e-4 --save_path models/sphere_18_20
python generate_from_diffusion_synthetic.py --model models/sphere_18_20 --intrinsic_dim 18 --ambient_dim 20 --num_samples 100000 --batch_size 256 --save_path "manifolds/sphere_18_20_t=20" --timestep 20

python train_diffusion_model_synthetic.py --intrinsic_dim 10 --ambient_dim 100 --num_points 100000 --num_epochs 30 --batch_size 64 --lr 4e-4 --save_path models/sphere_10_100
python generate_from_diffusion_synthetic.py --model models/sphere_10_100 --intrinsic_dim 10 --ambient_dim 100 --num_samples 100000 --batch_size 256 --save_path "manifolds/sphere_10_100_t=20" --timestep 20
python train_diffusion_model_synthetic.py --intrinsic_dim 50 --ambient_dim 100 --num_points 100000 --num_epochs 30 --batch_size 64 --lr 4e-4 --save_path models/sphere_50_100
python generate_from_diffusion_synthetic.py --model models/sphere_50_100 --intrinsic_dim 50 --ambient_dim 100 --num_samples 100000 --batch_size 256 --save_path "manifolds/sphere_50_100_t=20" --timestep 20
python train_diffusion_model_synthetic.py --intrinsic_dim 90 --ambient_dim 100 --num_points 100000 --num_epochs 30 --batch_size 64 --lr 4e-4 --save_path models/sphere_90_100
python generate_from_diffusion_synthetic.py --model models/sphere_90_100 --intrinsic_dim 90 --ambient_dim 100 --num_samples 100000 --batch_size 256 --save_path "manifolds/sphere_90_100_t=20" --timestep 20

#####################################################
# REAL WORLD DATASETS
#####################################################

# Train Diffusion Models
python train_diffusion_model.py --dataset mnist --image_size 12 --image_channels 1 --num_epochs 30 --batch_size 64 --lr 4e-4 --save_path models/mnist12
python train_diffusion_model.py --dataset mnist --image_size 28 --image_channels 1 --num_epochs 30 --batch_size 64 --lr 4e-4 --save_path models/mnist
python train_diffusion_model.py --dataset fashion_mnist --image_size 12 --image_channels 1 --num_epochs 30 --batch_size 64 --lr 4e-4 --save_path models/fmnist12
python train_diffusion_model.py --dataset fashion_mnist --image_size 28 --image_channels 1 --num_epochs 30 --batch_size 64 --lr 4e-4 --save_path models/fmnist
python train_diffusion_model.py --dataset datasets/kmnist --image_size 12 --image_channels 1 --num_epochs 30 --batch_size 64 --lr 4e-4 --save_path models/kmnist12
python train_diffusion_model.py --dataset datasets/kmnist --image_size 28 --image_channels 1 --num_epochs 30 --batch_size 64 --lr 4e-4 --save_path models/kmnist

# Intrinsic Dimension MNIST 28
python generate_from_diffusion.py --model models/mnist --dataset mnist --image_size 28 --x_0 label_0 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist_t=20_x=0" --timestep 20
python generate_from_diffusion.py --model models/mnist --dataset mnist --image_size 28 --x_0 label_1 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist_t=20_x=1" --timestep 20
python generate_from_diffusion.py --model models/mnist --dataset mnist --image_size 28 --x_0 label_2 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist_t=20_x=2" --timestep 20
python generate_from_diffusion.py --model models/mnist --dataset mnist --image_size 28 --x_0 label_3 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist_t=20_x=3" --timestep 20
python generate_from_diffusion.py --model models/mnist --dataset mnist --image_size 28 --x_0 label_4 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist_t=20_x=4" --timestep 20
python generate_from_diffusion.py --model models/mnist --dataset mnist --image_size 28 --x_0 label_5 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist_t=20_x=5" --timestep 20
python generate_from_diffusion.py --model models/mnist --dataset mnist --image_size 28 --x_0 label_6 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist_t=20_x=6" --timestep 20
python generate_from_diffusion.py --model models/mnist --dataset mnist --image_size 28 --x_0 label_7 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist_t=20_x=7" --timestep 20
python generate_from_diffusion.py --model models/mnist --dataset mnist --image_size 28 --x_0 label_8 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist_t=20_x=8" --timestep 20
python generate_from_diffusion.py --model models/mnist --dataset mnist --image_size 28 --x_0 label_9 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist_t=20_x=9" --timestep 20

# Intrinsic Dimension FMNIST 28
python generate_from_diffusion.py --model models/fmnist --dataset fashion_mnist --image_size 28 --x_0 label_0 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist_t=20_x=0" --timestep 20
python generate_from_diffusion.py --model models/fmnist --dataset fashion_mnist --image_size 28 --x_0 label_1 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist_t=5_x=1" --timestep 5
python generate_from_diffusion.py --model models/fmnist --dataset fashion_mnist --image_size 28 --x_0 label_2 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist_t=1_x=2" --timestep 1
python generate_from_diffusion.py --model models/fmnist --dataset fashion_mnist --image_size 28 --x_0 label_3 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist_t=5_x=3" --timestep 5
python generate_from_diffusion.py --model models/fmnist --dataset fashion_mnist --image_size 28 --x_0 label_4 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist_t=20_x=4" --timestep 20
python generate_from_diffusion.py --model models/fmnist --dataset fashion_mnist --image_size 28 --x_0 label_5 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist_t=20_x=5" --timestep 20
python generate_from_diffusion.py --model models/fmnist --dataset fashion_mnist --image_size 28 --x_0 label_6 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist_t=2_x=6" --timestep 2
python generate_from_diffusion.py --model models/fmnist --dataset fashion_mnist --image_size 28 --x_0 label_7 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist_t=20_x=7" --timestep 20
python generate_from_diffusion.py --model models/fmnist --dataset fashion_mnist --image_size 28 --x_0 label_8 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist_t=2_x=8" --timestep 2
python generate_from_diffusion.py --model models/fmnist --dataset fashion_mnist --image_size 28 --x_0 label_9 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist_t=2_x=9" --timestep 2

# Intrinsic Dimension KMNIST 28
python generate_from_diffusion.py --model models/kmnist --dataset datasets/kmnist --image_size 28 --x_0 label_0 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist_t=20_x=0" --timestep 20
python generate_from_diffusion.py --model models/kmnist --dataset datasets/kmnist --image_size 28 --x_0 label_1 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist_t=20_x=1" --timestep 20
python generate_from_diffusion.py --model models/kmnist --dataset datasets/kmnist --image_size 28 --x_0 label_2 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist_t=20_x=2" --timestep 20
python generate_from_diffusion.py --model models/kmnist --dataset datasets/kmnist --image_size 28 --x_0 label_3 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist_t=20_x=3" --timestep 20
python generate_from_diffusion.py --model models/kmnist --dataset datasets/kmnist --image_size 28 --x_0 label_4 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist_t=20_x=4" --timestep 20
python generate_from_diffusion.py --model models/kmnist --dataset datasets/kmnist --image_size 28 --x_0 label_5 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist_t=20_x=5" --timestep 20
python generate_from_diffusion.py --model models/kmnist --dataset datasets/kmnist --image_size 28 --x_0 label_6 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist_t=20_x=6" --timestep 20
python generate_from_diffusion.py --model models/kmnist --dataset datasets/kmnist --image_size 28 --x_0 label_7 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist_t=20_x=7" --timestep 20
python generate_from_diffusion.py --model models/kmnist --dataset datasets/kmnist --image_size 28 --x_0 label_8 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist_t=20_x=8" --timestep 20
python generate_from_diffusion.py --model models/kmnist --dataset datasets/kmnist --image_size 28 --x_0 label_9 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist_t=20_x=9" --timestep 20

# Intrinsic Dimension MNIST 12
python generate_from_diffusion.py --model models/mnist12 --dataset mnist --image_size 12 --x_0 label_0 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist12_t=5_x=0" --timestep 5
python generate_from_diffusion.py --model models/mnist12 --dataset mnist --image_size 12 --x_0 label_1 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist12_t=5_x=1" --timestep 5
python generate_from_diffusion.py --model models/mnist12 --dataset mnist --image_size 12 --x_0 label_2 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist12_t=5_x=2" --timestep 5
python generate_from_diffusion.py --model models/mnist12 --dataset mnist --image_size 12 --x_0 label_3 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist12_t=5_x=3" --timestep 5
python generate_from_diffusion.py --model models/mnist12 --dataset mnist --image_size 12 --x_0 label_4 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist12_t=5_x=4" --timestep 5
python generate_from_diffusion.py --model models/mnist12 --dataset mnist --image_size 12 --x_0 label_5 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist12_t=5_x=5" --timestep 5
python generate_from_diffusion.py --model models/mnist12 --dataset mnist --image_size 12 --x_0 label_6 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist12_t=5_x=6" --timestep 5
python generate_from_diffusion.py --model models/mnist12 --dataset mnist --image_size 12 --x_0 label_7 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist12_t=5_x=7" --timestep 5
python generate_from_diffusion.py --model models/mnist12 --dataset mnist --image_size 12 --x_0 label_8 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist12_t=5_x=8" --timestep 5
python generate_from_diffusion.py --model models/mnist12 --dataset mnist --image_size 12 --x_0 label_9 --num_samples 100000 --batch_size 256 --save_path "manifolds/mnist12_t=5_x=9" --timestep 5

# Intrinsic Dimension FMNIST 12
python generate_from_diffusion.py --model models/fmnist12 --dataset fashion_mnist --image_size 12 --x_0 label_0 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist12_t=5_x=0" --timestep 5
python generate_from_diffusion.py --model models/fmnist12 --dataset fashion_mnist --image_size 12 --x_0 label_1 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist12_t=5_x=1" --timestep 5
python generate_from_diffusion.py --model models/fmnist12 --dataset fashion_mnist --image_size 12 --x_0 label_2 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist12_t=5_x=2" --timestep 5
python generate_from_diffusion.py --model models/fmnist12 --dataset fashion_mnist --image_size 12 --x_0 label_3 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist12_t=5_x=3" --timestep 5
python generate_from_diffusion.py --model models/fmnist12 --dataset fashion_mnist --image_size 12 --x_0 label_4 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist12_t=5_x=4" --timestep 5
python generate_from_diffusion.py --model models/fmnist12 --dataset fashion_mnist --image_size 12 --x_0 label_5 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist12_t=5_x=5" --timestep 5
python generate_from_diffusion.py --model models/fmnist12 --dataset fashion_mnist --image_size 12 --x_0 label_6 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist12_t=5_x=6" --timestep 5
python generate_from_diffusion.py --model models/fmnist12 --dataset fashion_mnist --image_size 12 --x_0 label_7 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist12_t=5_x=7" --timestep 5
python generate_from_diffusion.py --model models/fmnist12 --dataset fashion_mnist --image_size 12 --x_0 label_8 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist12_t=5_x=8" --timestep 5
python generate_from_diffusion.py --model models/fmnist12 --dataset fashion_mnist --image_size 12 --x_0 label_9 --num_samples 100000 --batch_size 256 --save_path "manifolds/fmnist12_t=5_x=9" --timestep 5

# Intrinsic Dimension KMNIST 12
python generate_from_diffusion.py --model models/kmnist12 --dataset datasets/kmnist --image_size 12 --x_0 label_0 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist12_t=5_x=0" --timestep 5
python generate_from_diffusion.py --model models/kmnist12 --dataset datasets/kmnist --image_size 12 --x_0 label_1 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist12_t=5_x=1" --timestep 5
python generate_from_diffusion.py --model models/kmnist12 --dataset datasets/kmnist --image_size 12 --x_0 label_2 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist12_t=5_x=2" --timestep 5
python generate_from_diffusion.py --model models/kmnist12 --dataset datasets/kmnist --image_size 12 --x_0 label_3 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist12_t=5_x=3" --timestep 5
python generate_from_diffusion.py --model models/kmnist12 --dataset datasets/kmnist --image_size 12 --x_0 label_4 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist12_t=5_x=4" --timestep 5
python generate_from_diffusion.py --model models/kmnist12 --dataset datasets/kmnist --image_size 12 --x_0 label_5 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist12_t=5_x=5" --timestep 5
python generate_from_diffusion.py --model models/kmnist12 --dataset datasets/kmnist --image_size 12 --x_0 label_6 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist12_t=5_x=6" --timestep 5
python generate_from_diffusion.py --model models/kmnist12 --dataset datasets/kmnist --image_size 12 --x_0 label_7 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist12_t=5_x=7" --timestep 5
python generate_from_diffusion.py --model models/kmnist12 --dataset datasets/kmnist --image_size 12 --x_0 label_8 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist12_t=5_x=8" --timestep 5
python generate_from_diffusion.py --model models/kmnist12 --dataset datasets/kmnist --image_size 12 --x_0 label_9 --num_samples 100000 --batch_size 256 --save_path "manifolds/kmnist12_t=5_x=9" --timestep 5
