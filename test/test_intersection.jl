using Test

include("../src/lib.jl")
using .GraphicsCore.Intersections
using .GraphicsCore.GeometryTypes

@testset verbose = true "Intersection Tests" begin
  @testset "Triangle Intersections" begin
    ray = Ray([0, 0, 0], [1, 0, 0])
    tri = Triangle([-2, -1, -1], [2, 1, -1], [2, 0, 1])

    @test ray.origin ≈ [0, 0, 0]

    @test intersect_test(tri, ray)

    ray = Ray([0, 0, 0], [1, 0, 0])
    tri = Triangle([-2, -1, -1], [-2, 1, -1], [-2, 0, 1])

    @test intersect_test(tri, ray)

    ray = Ray([0, 0, 0], [-1, 0, 0])
    tri = Triangle([-2, -1, -1], [-2, 1, -1], [-2, 0, 1])

    @test intersect_test(tri, ray)

    ray = Ray([0, 0, 0], [-1, 0, 0])
    tri = Triangle([2, -1, -1], [2, 1, -1], [2, 0, 1])

    @test intersect_test(tri, ray)
  end

end