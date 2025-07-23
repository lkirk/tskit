{
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";

  outputs =
    { self, nixpkgs }@inputs:
    let
      system = "x86_64-linux";
      pkgs = import inputs.nixpkgs { inherit system; };
    in
    (pkgs.buildFHSEnv {
      name = "python";
      targetPkgs = pkgs: [
        pkgs.python311Full
        pkgs.gcc
        pkgs.pkg-config
      ];
      runScript = "bash";
    }).env;
}
