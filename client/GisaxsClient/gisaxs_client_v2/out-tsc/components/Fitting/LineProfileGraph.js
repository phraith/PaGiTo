var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
import { jsx as _jsx } from "react/jsx-runtime";
import { Box } from '@mui/material';
import { LineCanvas } from '@nivo/line';
import React from 'react';
var LineProfileGraph = function (props) {
    console.log(props.plotData);
    return (_jsx(Box, __assign({ sx: {
            top: 0,
            paddingTop: 10,
            paddingRight: 5,
            paddingLeft: 10,
        } }, { children: _jsx(LineCanvas, { width: 2000, height: 600, data: props.plotData, margin: { top: 50, right: 160, bottom: 50, left: 60 }, axisTop: null, axisRight: null, axisLeft: {
                tickSize: 4,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'y',
                legendOffset: -41,
                legendPosition: 'middle'
            }, axisBottom: {
                tickSize: 0,
                tickPadding: 5,
                tickRotation: 0,
                legend: 'x',
                legendOffset: 36,
                legendPosition: 'middle'
            }, enablePoints: false, enableGridX: false, enableGridY: false, xScale: {
                type: 'linear',
                min: 'auto',
                max: 'auto'
            }, yScale: {
                type: 'linear',
                min: 'auto',
                max: 'auto'
            } }) })));
};
export default React.memo(LineProfileGraph);
