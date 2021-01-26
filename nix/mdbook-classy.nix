{ stdenv, fetchFromGitHub, rustPlatform }:

rustPlatform.buildRustPackage rec {
  name = "mdbook-classy";

  src = fetchFromGitHub {
    owner = "Reviewable";
    repo = "mdbook-classy";
    rev = "577bbbeee0429445fa0f35b07fd7517d6c8bd5c5";
    sha256 = "177j733lw7zw52fni5vrmzlnydwd2wkrclv6khmqgw0z6c6paxmp";
  };

  cargoSha256 = "10f85r75dlwavk3rck1x61ck0l9brf3y1ffmrqi55gbiqm509xmw";
}
