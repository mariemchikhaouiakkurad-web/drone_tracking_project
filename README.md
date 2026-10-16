# DALBD

Shared base. Each part of the project lives in its own branch, created from `main`:

| Branch | Content |
|---|---|
| `pipeline_sound` | acoustic pipeline: UMA-8 -> ODAS -> known-source drop (track id + direction) -> MobileNetV2 -> localisation, RViz, warning |

Add your part:
    git clone https://github.com/mariemchikhaouiakkurad-web/DALBD.git && cd DALBD
    git checkout -b <your_branch> && git add . && git commit -m "..." && git push -u origin <your_branch>
