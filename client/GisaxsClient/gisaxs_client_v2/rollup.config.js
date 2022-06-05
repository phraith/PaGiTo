"use strict";


import scss from 'rollup-plugin-scss'
import replace from "@rollup/plugin-replace";
import copy from 'rollup-plugin-copy'
import commonjs from '@rollup/plugin-commonjs';
import { babel } from '@rollup/plugin-babel';
import { nodeResolve } from '@rollup/plugin-node-resolve';
import { visualizer } from "rollup-plugin-visualizer";
import { resolve } from "path";
import serve from 'rollup-plugin-serve'
import livereload from 'rollup-plugin-livereload'
import path from 'path';


export default {

    inlineDynamicImports: true,
    input: 'out-tsc/index.js',
    output: {
        indent: false,
        file: 'dist/main.js',
        format: 'umd'
    },
    cache: true,

    plugins: [
        nodeResolve({
            extensions: [".js"],
        }),
        commonjs({ sourceMap: false }),
        // babel({
        //     extensions: [".ts"],
        //     include: resolve("src", "**", "*.ts"),
        //     babelHelpers: 'bundle'
        // }),
        scss(),
        replace({
            preventAssignment: true,
            'process.env.NODE_ENV': JSON.stringify(
                process.env.NODE_ENV
            )
        }),
        copy({
            targets: [
                { src: 'index.html', dest: 'dist' },
            ]
        }),
        visualizer(),
        serve(
        {
            historyApiFallback: true,
            contentBase: 'dist',
            open: false,
            host: 'localhost',
            port: 9996,
        }),
        livereload()
    ]
};