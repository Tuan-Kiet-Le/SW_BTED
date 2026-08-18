# SW-BTED bilingual review site

Static, self-contained lecturer review page with English/Vietnamese toggle.

## Local preview

From the project root:

```powershell
python -m http.server 8000 --directory review-site
```

Open <http://localhost:8000>.

## Vercel deployment

Set the Vercel project Root Directory to `review-site`, Framework Preset to
`Other`, and leave the build command empty. The site is a static
`index.html`; no build step is required.
