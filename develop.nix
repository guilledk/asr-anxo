with (import <nixpkgs> {});
with python3Packages;
stdenv.mkDerivation {
  name = "pip-env";
  buildInputs = [
    # System requirements.
    readline
    ffmpeg

    # Python requirements (enough to get a virtualenv going).
    python3Full
    virtualenv
    setuptools
    pip

    tqdm
    numpy
    torch
    torchvision
    torchaudio
    transformers
    datasets
    librosa
    soundfile
    pydub
    pysrt
  ];
  src = null;
  shellHook = ''
    # Allow the use of wheels.
    SOURCE_DATE_EPOCH=$(date +%s)

    if [ ! -d "venv" ]; then
        virtualenv venv
    fi

    source venv/bin/activate
  '';
}
