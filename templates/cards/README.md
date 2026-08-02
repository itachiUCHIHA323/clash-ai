# Card Icon Templates

Place 30x30 or 40x40 pixel PNG snippets of deployment card icons in this directory.

Supported filenames (case-insensitive):
- `SNEAKY_GOBLIN.png`
- `VALKYRIE.png`
- `DRAGON.png`
- `EDRAGON.png`
- `KING.png`
- `QUEEN.png`
- `WARDEN.png`
- `CHAMPION.png`

When these templates are present, `CardScanner` will automatically use OpenCV template matching to locate cards regardless of their slot order. If no templates are present, it falls back to the default 1280x720 slot coordinate mapping.
