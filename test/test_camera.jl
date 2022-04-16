using Test

include("../src/camera.jl")
include("../src/geotypes.jl")
using .Cameras
using .GeometryTypes

@testset verbose = true "Camera Tests" begin
  @testset "Perspective Camera" begin
    cam = PerspectiveCamera(1, π / 4, [1, 2.25, 4], [-0.5, -0.75, -1])
    ray = get_ray(cam, 0.3, 0.3)

    @test isapprox(cam.horizontal, [0.793489, 0, -0.238047])
    @test isapprox(cam.vertical, [-0.118612, 0.718263, -0.395374])
    @test isapprox(cam.lower_left, [0.413425, 1.3926, 3.48626])
    @test isapprox(cam.position, [1, 2.25, 4])

    @test isapprox(ray.origin, cam.position)
    @test isapprox(ray.direction, [-0.373983, -0.624998, -0.685212])

    cam = PerspectiveCamera(2, π / 2, [100, 150, 999], [29, 16, 32])
    ray = get_ray(cam, 0.8, 0.6)

    @test isapprox(cam.horizontal, [3.98926, 0, -0.292903])
    @test isapprox(cam.vertical, [-0.0200492, 1.98117, -0.273064])
    @test isapprox(cam.lower_left, [97.9429, 148.873, 998.295])
    @test isapprox(cam.position, [100, 150, 999])

    @test isapprox(ray.origin, cam.position)
    @test isapprox(ray.direction, [0.712626, 0.0388739, -0.700466])
  end

  @testset "Orthographic Camera" begin
    cam = OrthographicCamera([10, 9, 11], [1, 0, 0], [0, 1, 0], [-61, 9, 11])
    ray_corner = get_ray(cam, 0, 0)
    ray_center = get_ray(cam, 0.5, 0.5)
    ray_opposite = get_ray(cam, 1, 1)

    @test isapprox(cam.center, [10, 9, 11])
    @test isapprox(cam.direction, [-1, 0, 0])
    @test isapprox(cam.horizontal, [1, 0, 0])
    @test isapprox(cam.vertical, [0, 1, 0])

    @test isapprox(ray_corner.origin, [9.5, 8.5, 11])
    @test isapprox(ray_center.origin, [10, 9, 11])
    @test isapprox(ray_opposite.origin, [10.5, 9.5, 11])

    @test isapprox(ray_corner.direction, [-1, 0, 0])
    @test isapprox(ray_center.direction, [-1, 0, 0])
    @test isapprox(ray_opposite.direction, [-1, 0, 0])
  end

end