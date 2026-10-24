import { mdsvex } from 'mdsvex';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: [vitePreprocess(), mdsvex()],

	kit: {
		paths: {
			relative: true
		},
		router: { type: 'hash' },
		adapter: adapter({
			pages: '../public',
			assets: '../public',
			fallback: 'index.html',
			precompress: false,
			strict: true
		}),
		output: {
			bundleStrategy: 'inline'
		},
		alias: {
			$styles: 'src/styles'
		},
		// A fixed name keeps the tracked ../public/index.html.gz reproducible,
		// so installing from a clean clone leaves the checkout unchanged.
		version: {
			name: 'slstack'
		}
	},

	extensions: ['.svelte', '.svx']
};

export default config;
