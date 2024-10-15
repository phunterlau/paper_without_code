import { Transformer } from 'markmap-lib';
import { fillTemplate } from 'markmap-render';
import nodeHtmlToImage from 'node-html-to-image';
import { readFile, writeFile } from 'node:fs/promises';

async function renderMarkmap(markdown, outFile) {
    const transformer = new Transformer();
    const { root, features } = transformer.transform(markdown);
    const assets = transformer.getUsedAssets(features);
    const html =
        fillTemplate(root, assets, {
            jsonOptions: {
                duration: 0,
                maxInitialScale: 5,
            },
        }) +
        `
<style>
body,
#mindmap {
  width: 2400px;
  height: 1800px;
}
</style>
`;
    const image = await nodeHtmlToImage({
        html,
    });
    await writeFile(outFile, image);
}

async function main() {
    if (process.argv.length < 4) {
        console.log('Usage: node script.js <input_markdown_file> <output_png_file>');
        process.exit(1);
    }

    const inputFile = process.argv[2];
    const outputFile = process.argv[3];

    try {
        const markdown = await readFile(inputFile, 'utf-8');
        await renderMarkmap(markdown, outputFile);
        console.log(`Mindmap generated successfully: ${outputFile}`);
    } catch (error) {
        console.error('Error:', error.message);
        process.exit(1);
    }
}

main();
