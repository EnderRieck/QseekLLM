#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import MarkdownIt from "markdown-it";
import { chromium } from "playwright";

const args = process.argv.slice(2);

function usage() {
  console.error(
    [
      "Usage:",
      "  node scripts/render_report_pdf.mjs <input.md> <output.pdf> [--html <output.html>] [--css <report.css>]",
      "",
      "Example:",
      "  node scripts/render_report_pdf.mjs docs/中期报告.md docs/中期报告.pdf",
    ].join("\n"),
  );
}

function argValue(name) {
  const index = args.indexOf(name);
  if (index === -1) return null;
  return args[index + 1] ?? null;
}

if (args.length < 2 || args.includes("-h") || args.includes("--help")) {
  usage();
  process.exit(args.includes("-h") || args.includes("--help") ? 0 : 1);
}

const rootDir = process.cwd();
const inputPath = path.resolve(rootDir, args[0]);
const outputPdfPath = path.resolve(rootDir, args[1]);
const outputHtmlPath = path.resolve(
  rootDir,
  argValue("--html") ?? args[1].replace(/\.pdf$/i, ".html"),
);
const cssPath = path.resolve(rootDir, argValue("--css") ?? "docs/report.css");

const markdown = await fs.readFile(inputPath, "utf8");
const css = await fs.readFile(cssPath, "utf8");

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
});

const defaultImageRenderer =
  md.renderer.rules.image ??
  ((tokens, idx, options, env, self) => self.renderToken(tokens, idx, options));

md.renderer.rules.image = (tokens, idx, options, env, self) => {
  const token = tokens[idx];
  const alt = token.content || token.attrGet("alt") || "";
  const rendered = defaultImageRenderer(tokens, idx, options, env, self);
  const caption = alt
    ? `<figcaption>${md.utils.escapeHtml(alt)}</figcaption>`
    : "";
  return `<figure class="figure">${rendered}${caption}</figure>`;
};

const renderedMarkdown = md
  .render(markdown)
  .replaceAll("<p><figure", "<figure")
  .replaceAll("</figure></p>", "</figure>");
const headings = collectHeadings(renderedMarkdown);
const title = extractTitle(markdown) ?? "报告";
const date = extractDate(markdown);
const relativeCssPath = path.relative(path.dirname(outputHtmlPath), cssPath);

const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(title)}</title>
  <style>${css}</style>
  <link rel="stylesheet" href="${toPosix(relativeCssPath)}">
</head>
<body>
  <main class="page report">
    <section class="cover">
      <p class="cover-kicker">郭子逸 · 张子路｜「千语千寻」大模型训练项目</p>
      <h1>${escapeHtml(title)}</h1>
      ${date ? `<p class="cover-meta">${escapeHtml(date)}</p>` : ""}
    </section>
    <section class="toc">
      <h2>目录</h2>
      <ol>
        ${headings
          .filter((heading) => heading.level === 2)
          .map(
            (heading) =>
              `<li><a href="#${heading.id}">${escapeHtml(heading.text)}</a></li>`,
          )
          .join("\n        ")}
      </ol>
    </section>
    <article class="markdown-body">
      ${withHeadingIds(renderedMarkdown, headings)}
    </article>
  </main>
</body>
</html>`;

await fs.mkdir(path.dirname(outputHtmlPath), { recursive: true });
await fs.mkdir(path.dirname(outputPdfPath), { recursive: true });
await fs.writeFile(outputHtmlPath, html, "utf8");

let browser;
try {
  browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({
    viewport: { width: 1240, height: 1754 },
    deviceScaleFactor: 1,
  });
  await page.goto(pathToFileURL(outputHtmlPath).href, {
    waitUntil: "networkidle",
  });
  await page.pdf({
    path: outputPdfPath,
    format: "A4",
    printBackground: true,
    preferCSSPageSize: true,
    displayHeaderFooter: false,
  });
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes("Executable doesn't exist")) {
    console.error(
      "Playwright Chromium is not installed. Run: npx playwright install chromium",
    );
  }
  throw error;
} finally {
  if (browser) {
    await browser.close();
  }
}

console.log(`HTML written to ${path.relative(rootDir, outputHtmlPath)}`);
console.log(`PDF written to ${path.relative(rootDir, outputPdfPath)}`);

function extractTitle(source) {
  const match = source.match(/^#\s+(.+)$/m);
  return match?.[1]?.trim();
}

function extractDate(source) {
  const match = source.match(/^日期[:：]\s*(.+)$/m);
  return match?.[1]?.trim();
}

function collectHeadings(sourceHtml) {
  const headings = [];
  const re = /<h([2-4])>(.*?)<\/h\1>/g;
  let match;
  const used = new Map();
  while ((match = re.exec(sourceHtml)) !== null) {
    const level = Number(match[1]);
    const text = stripTags(match[2]).trim();
    const base = slugify(text);
    const count = used.get(base) ?? 0;
    used.set(base, count + 1);
    const id = count === 0 ? base : `${base}-${count + 1}`;
    headings.push({ level, text, id, raw: match[0] });
  }
  return headings;
}

function withHeadingIds(sourceHtml, headings) {
  let output = sourceHtml;
  for (const heading of headings) {
    const replacement = heading.raw.replace(
      /^<h([2-4])>/,
      `<h$1 id="${heading.id}">`,
    );
    output = output.replace(heading.raw, replacement);
  }
  return output;
}

function stripTags(input) {
  return input.replace(/<[^>]+>/g, "");
}

function slugify(input) {
  const normalized = input
    .toLowerCase()
    .replace(/&amp;/g, "and")
    .replace(/[^\p{Script=Han}\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-+|-+$/g, "");
  return normalized || "section";
}

function escapeHtml(input) {
  return input
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function toPosix(input) {
  return input.split(path.sep).join("/");
}
