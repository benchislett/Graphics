using Test

using .GraphicsCore.Cameras
using .GraphicsCore.GeometryPrimitives

@testset verbose = true "Camera Tests" begin
  @testset "Perspective Camera" begin
    cam = PerspectiveCamera(1, π / 4, [1, 2.25, 4], [-0.5, -0.75, -1])
    ray = get_ray(cam, 0.3, 0.3)

    @test cam.horizontal ≈ [0.793489, 0, -0.238047]
    @test cam.vertical ≈ [-0.118612, 0.718263, -0.395374]
    @test cam.lower_left ≈ [0.413425, 1.3926, 3.48626]
    @test cam.position ≈ [1, 2.25, 4]

    @test ray.origin ≈ cam.position
    @test ray.direction ≈ [-0.373983, -0.624998, -0.685212]

    cam = PerspectiveCamera(2, π / 2, [100, 150, 999], [29, 16, 32])
    ray = get_ray(cam, 0.8, 0.6)

    @test cam.horizontal ≈ [3.98926, 0, -0.292903]
    @test cam.vertical ≈ [-0.0200492, 1.98117, -0.273064]
    @test cam.lower_left ≈ [97.9429, 148.873, 998.295]
    @test cam.position ≈ [100, 150, 999]

    @test ray.origin ≈ cam.position
    @test ray.direction ≈ [0.712626, 0.0388739, -0.700466]
  end

  @testset "Orthographic Camera" begin
    cam = OrthographicCamera([10, 9, 11], [1, 0, 0], [0, 1, 0], [-61, 9, 11])
    ray_corner = get_ray(cam, 0, 0)
    ray_center = get_ray(cam, 0.5, 0.5)
    ray_opposite = get_ray(cam, 1, 1)

    @test cam.center ≈ [10, 9, 11]
    @test cam.direction ≈ [-1, 0, 0]
    @test cam.horizontal ≈ [1, 0, 0]
    @test cam.vertical ≈ [0, 1, 0]

    @test ray_corner.origin ≈ [9.5, 8.5, 11]
    @test ray_center.origin ≈ [10, 9, 11]
    @test ray_opposite.origin ≈ [10.5, 9.5, 11]

    @test ray_corner.direction ≈ [-1, 0, 0]
    @test ray_center.direction ≈ [-1, 0, 0]
    @test ray_opposite.direction ≈ [-1, 0, 0]
  end

end