# starcraft:game + fixes for scbw --auto_launch (headful lobby creation):
# - xdotool is missing from the base image.
# - auto_launch clicks Start after only 3 s, before the opponent bot joins
#   (game starts 1-player, bots crash on enemy() == NULL). Wait longer.
FROM starcraft:game
USER root
RUN apt-get -o Dir::Etc::SourceParts=/nonexistent update \
    && apt-get -o Dir::Etc::SourceParts=/nonexistent install -y --no-install-recommends xdotool \
    && rm -rf /var/lib/apt/lists/*
RUN grep -q '^    sleep 3$' /app/play_common.sh \
    && sed -i 's/^    sleep 3$/    sleep 20/' /app/play_common.sh
USER starcraft
