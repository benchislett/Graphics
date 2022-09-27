module Materials

export Material, DefaultMaterial

abstract type Material end

struct DefaultMaterial <: Material end

end