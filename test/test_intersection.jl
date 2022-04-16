using Test

using .GraphicsCore.Intersections
using .GraphicsCore.GeometryTypes

@testset verbose = true "Intersection Tests" begin
  @testset "Triangle Intersections" begin
    ray = Ray([0, 0, 0], [1, 0, 0])
    tri = Triangle([2, -1, -1], [2, 1, -1], [2, 0, 1])

    @test ray.origin ≈ [0, 0, 0]

    @test intersect_test(tri, ray)
    @test intersection(tri, ray).time ≈ 2
    @test intersection(tri, ray).uvw ≈ [0.25, 0.5, 0.25]

    ray = Ray([0, 0, 0], [1, 0, 0])
    tri = Triangle([-2, -1, -1], [-2, 1, -1], [-2, 0, 1])

    @test !intersect_test(tri, ray)

    ray = Ray([0, 0, 0], [-1, 0, 0])
    tri = Triangle([-2, -1, -1], [-2, 1, -1], [-2, 0, 1])

    @test intersect_test(tri, ray)
    @test intersection(tri, ray).time ≈ 2
    @test intersection(tri, ray).uvw ≈ [0.25, 0.5, 0.25]

    ray = Ray([0, 0, 0], [-1, 0, 0])
    tri = Triangle([2, -1, -1], [2, 1, -1], [2, 0, 1])

    @test !intersect_test(tri, ray)
  end

  @testset "Sphere Intersections" begin
    sphere = Sphere([2, 3, 4], 4.9)
    ray = Ray([8, 0, 0], [-1, 0, 0])

    @test !intersect_test(sphere, ray)

    sphere = Sphere([2, 3, 4], 6)

    @test intersect_test(sphere, ray)
    @test intersection(sphere, ray).time ≈ 2.68337
  end

end