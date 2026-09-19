# LightToken execution branch note

Implementation is executed on a dedicated feature branch created from the approved design/plan head. The GitHub connector in this session does not expose a local checkout/worktree filesystem, so branch isolation is the harness-native equivalent used here. `main` remains untouched until full PR verification succeeds.
