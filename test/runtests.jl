module Tests

using Test

include("../src/GraphicsCore.jl")

@testset verbose = true "Main" begin
  include("test_camera.jl")
  include("test_intersection.jl")
end

end