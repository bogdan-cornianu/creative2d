import Phaser from "phaser";
import { useEffect, useRef } from "react";
import { Manifest } from "./api";

interface Props {
  manifest: Manifest;
  zoom: number;
  pixelArt: boolean;
}

/** Loads the exported files exactly as a Phaser game would and shows them in use. */
export function PhaserPreview({ manifest, zoom, pixelArt }: Props) {
  const host = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const r = manifest.result;
    const base = `/outputs/${manifest.job_id}/assets/`;

    class PreviewScene extends Phaser.Scene {
      preload() {
        this.load.setPath(base);
        for (const e of r.entries) {
          if (e.loader === "atlas") this.load.atlas(e.key, e.texture!, e.json!);
          else if (e.loader === "multiatlas") this.load.multiatlas(e.key, e.json!, base);
          else if (e.loader === "tileset")
            this.load.spritesheet(e.key, e.texture!, {
              frameWidth: e.tileWidth!,
              frameHeight: e.tileHeight!,
              margin: e.margin,
              spacing: e.spacing,
            });
          else if (e.loader === "image") this.load.image(e.key, e.texture!);
        }
        const animFile = r.anim_files?.find((f) => !f.includes("_sheet_"));
        if (animFile) this.load.json("anims", animFile);
      }

      create() {
        const cam = this.cameras.main;
        cam.setZoom(zoom);
        if (r.kind === "sprites") this.sprites();
        else if (r.kind === "tiles") this.tiles();
        else this.background();
      }

      sprites() {
        const atlas = r.entries.find((e) => e.loader === "atlas" || e.loader === "multiatlas");
        if (!atlas) return;
        const data = this.cache.json.get("anims");
        if (data) this.anims.fromJSON(data);
        const size = r.frame_size ?? 64;
        const animated = new Set((r.anims ?? []).flatMap((a) => a.frames));
        const items: { frame: string; anim?: string }[] = [
          ...(r.anims ?? []).map((a) => ({ frame: a.frames[0], anim: a.key })),
          ...(r.frames ?? []).filter((f) => !animated.has(f)).map((f) => ({ frame: f })),
        ];
        const gap = size * 0.4;
        const cols = Math.max(1, Math.min(items.length, Math.floor(this.scale.width / zoom / (size + gap))));
        const rows = Math.ceil(items.length / cols);
        const w = cols * size + (cols - 1) * gap;
        const h = rows * size + (rows - 1) * gap;
        items.forEach((it, i) => {
          const x = (i % cols) * (size + gap) + size / 2;
          const y = Math.floor(i / cols) * (size + gap) + size;
          const s = this.add.sprite(x, y, atlas.key, it.frame).setOrigin(0.5, 1);
          if (it.anim) s.play(it.anim);
          // Ground line shows the anchor the frames share.
          this.add.rectangle(x, y, size, 1, 0x2f5bea, 0.35).setOrigin(0.5, 0);
        });
        this.cameras.main.centerOn(w / 2, h / 2);
      }

      tiles() {
        const sheet = r.entries.find((e) => e.loader === "tileset");
        const atlas = r.entries.find((e) => e.loader === "atlas");
        const count = r.frames?.length ?? 1;
        const tw = r.tile_width ?? 32;
        const n = Math.max(8, (r.chunk ?? 1) * 4);
        // Tiles come in chunk x chunk blocks per variant; keep blocks intact, vary variants per block.
        const k = r.chunk ?? 1;
        const variants = Math.max(1, Math.floor(count / (k * k)));
        const pick = (c: number, row: number) => {
          const bc = Math.floor(c / k);
          const br = Math.floor(row / k);
          const v = (bc * 7 + br * 13 + ((bc * br) % 5)) % variants;
          return v * k * k + (row % k) * k + (c % k);
        };
        const place = (x: number, y: number, idx: number) =>
          sheet
            ? this.add.image(x, y, sheet.key, idx)
            : this.add.image(x, y, atlas!.key, r.frames![idx]);
        if (r.orientation === "isometric") {
          const hh = tw / 4; // half of the diamond height
          for (let row = 0; row < n; row++)
            for (let c = 0; c < n; c++)
              place((c - row) * (tw / 2), (c + row) * hh, pick(c, row)).setOrigin(0.5, 0).setDepth(c + row);
          this.cameras.main.centerOn(0, n * hh + (r.iso_depth ?? 0) / 2);
        } else {
          for (let row = 0; row < n; row++)
            for (let c = 0; c < n; c++) place(c * tw, row * tw, pick(c, row)).setOrigin(0, 0);
          this.cameras.main.centerOn((n * tw) / 2, (n * tw) / 2);
        }
      }

      background() {
        const layers = r.entries.filter((e) => e.loader === "image");
        const w = r.width ?? 1024;
        const h = r.height ?? 576;
        const sprites = layers.map((l) =>
          this.add.tileSprite(0, 0, w, h, l.key).setOrigin(0, 0).setScrollFactor(0),
        );
        const cam = this.cameras.main;
        cam.setZoom(Math.min(this.scale.width / w, this.scale.height / h) * zoom);
        cam.centerOn(w / 2, h / 2);
        this.events.on("update", (_t: number, dt: number) => {
          sprites.forEach((s, i) => {
            const factor = layers[i].scrollFactor ?? 1;
            s.tilePositionX += (dt / 1000) * 40 * factor;
          });
        });
      }
    }

    const game = new Phaser.Game({
      type: Phaser.AUTO,
      parent: el,
      transparent: true,
      pixelArt,
      scale: { mode: Phaser.Scale.RESIZE, width: el.clientWidth, height: el.clientHeight },
      scene: PreviewScene,
      banner: false,
      audio: { noAudio: true },
    });
    return () => game.destroy(true);
  }, [manifest, zoom, pixelArt]);

  return <div className="phaser-host" ref={host} aria-label="Live Phaser preview of the exported assets" />;
}
