module Tests

using Test

include("../src/lib.jl")

@testset verbose = true "Main" begin
  include("test_camera.jl")
  include("test_intersection.jl")
end

end