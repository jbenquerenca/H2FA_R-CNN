docker run \
      --name h2fa_r-cnn-container \
      --gpus all \
      -it \
      -e "color_prompt=yes" \
      -t \
      --shm-size 32g \
      --rm \
      -v $HOME/dissertacao/models/H2FA_R-CNN:/local \
      -v $HOME/dissertacao/data:/data \
      h2fa_r-cnn
