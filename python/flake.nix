{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  outputs =
    { self, nixpkgs }@inputs:
    let
      system = "x86_64-linux";
      pkgs = import inputs.nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default =
        (pkgs.buildFHSEnv {
          name = "python";
          targetPkgs = pkgs: [
            pkgs.gcc
	    pkgs.gsl
	    pkgs.gsl.dev
            pkgs.libz
	    pkgs.meson
	    pkgs.ninja
            pkgs.pkg-config
            pkgs.python313Full
          ];
          runScript = "fish";
        }).env;
    };
}
