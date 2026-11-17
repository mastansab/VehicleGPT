# VehicleAI web UI

Optional React chat surface that talks to the local vehicle server.

```
Browser
 ├─ chat agent (local / Workers)
 └─ live vehicle stream from `peykan demo`
```

## Local development

```bash
cd web
npm install
npm run dev
```

Start the vehicle process separately:

```bash
peykan demo --port 6278
```
