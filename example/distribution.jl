using Plots, Images, ImageView, ImageFiltering, LinearAlgebra

using GraphicsCore.GeometryTypes
using GraphicsCore.Sampling

N = 100
samples = [(rand(Scalar), rand(Scalar)) for i in 1:N]
@time samples = [sample_oriented_hemisphere(normalize(Vector3f(1, 1, 1)), u, v) for (u, v) in samples]
xs = [samples[i][1] for i in 1:N]
ys = [samples[i][2] for i in 1:N]
zs = [samples[i][3] for i in 1:N]

plotlyjs()
scatter3d(xs, ys, zs)