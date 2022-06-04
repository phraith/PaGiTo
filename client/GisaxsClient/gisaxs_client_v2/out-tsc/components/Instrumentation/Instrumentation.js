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
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import Card from "@mui/material/Card";
import CardActions from "@mui/material/CardActions";
import CardContent from "@mui/material/CardContent";
import InputAdornment from "@mui/material/InputAdornment";
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import React, { useEffect } from "react";
import { InstrumentationConfig, SetLocalStorageEntity, } from "../Utility/DefaultConfigs";
var Instrumentation = function (props) {
    var _a = React.useState(InstrumentationConfig.beam.alphai), alphaI = _a[0], setAlphaI = _a[1];
    var _b = React.useState(InstrumentationConfig.beam.photonEv), photonEv = _b[0], setPhotonEv = _b[1];
    var _c = React.useState(InstrumentationConfig.detector.beamImpact.x), beamX = _c[0], setBeamX = _c[1];
    var _d = React.useState(InstrumentationConfig.detector.beamImpact.y), beamY = _d[0], setBeamY = _d[1];
    var _e = React.useState(InstrumentationConfig.detector.resolution.width), resX = _e[0], setResX = _e[1];
    var _f = React.useState(InstrumentationConfig.detector.resolution.height), resY = _f[0], setResY = _f[1];
    var _g = React.useState(InstrumentationConfig.detector.pixelsize), pixelsize = _g[0], setPixelsize = _g[1];
    var _h = React.useState(InstrumentationConfig.detector.sampleDistance), sampleDistance = _h[0], setSampleDistance = _h[1];
    var localStorageEntityName = "instrumentationConfig";
    var configFieldName = "instrumentation";
    useEffect(function () {
        var currentConfig = {
            beam: {
                alphai: alphaI,
                photonEv: photonEv,
            },
            detector: {
                pixelsize: pixelsize,
                resolution: {
                    width: resX,
                    height: resY,
                },
                sampleDistance: sampleDistance,
                beamImpact: {
                    x: beamX,
                    y: beamY,
                },
            },
        };
        SetLocalStorageEntity(currentConfig, InstrumentationConfig, localStorageEntityName);
        props.jsonCallback(currentConfig, configFieldName);
    }, [alphaI, photonEv, beamX, beamY, resX, resY, pixelsize, sampleDistance]);
    useEffect(function () {
        var data = localStorage.getItem(localStorageEntityName);
        if (data !== null) {
            var instrumentationConfig = JSON.parse(data);
            setAlphaI(instrumentationConfig.beam.alphai);
            setPhotonEv(instrumentationConfig.beam.photonEv);
            setBeamX(instrumentationConfig.detector.beamImpact.x);
            setBeamY(instrumentationConfig.detector.beamImpact.y);
            setResX(instrumentationConfig.detector.resolution.width);
            setResY(instrumentationConfig.detector.resolution.height);
            setPixelsize(instrumentationConfig.detector.pixelsize);
            setSampleDistance(instrumentationConfig.detector.sampleDistance);
        }
    }, []);
    return (_jsxs(Card, __assign({ sx: {} }, { children: [_jsxs(CardContent, { children: [_jsx(Typography, { children: "Instrumentation" }), _jsxs(Grid, __assign({ container: true, sx: { paddingTop: 2 }, rowSpacing: 2 }, { children: [_jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(TextField, { InputProps: {
                                        startAdornment: (_jsx(InputAdornment, __assign({ position: "start" }, { children: "\u00B0" }))),
                                    }, label: "alphaI", value: alphaI, type: "number", onChange: function (e) {
                                        setAlphaI(Number(e.target.value));
                                    } }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(TextField, { InputProps: {
                                        startAdornment: (_jsx(InputAdornment, __assign({ position: "start" }, { children: "eV" }))),
                                    }, label: "photonEv", value: photonEv, type: "number", onChange: function (e) {
                                        setPhotonEv(Number(e.target.value));
                                    } }) })), _jsx(Grid, __assign({ item: true, xs: 3 }, { children: _jsx(TextField, { label: "beamX", value: beamX, type: "number", onChange: function (e) {
                                        setBeamX(Number(e.target.value));
                                    } }) })), _jsx(Grid, __assign({ item: true, xs: 3 }, { children: _jsx(TextField, { label: "beamY", value: beamY, type: "number", onChange: function (e) {
                                        setBeamY(Number(e.target.value));
                                    } }) })), _jsx(Grid, __assign({ item: true, xs: 3 }, { children: _jsx(TextField, { label: "resX", value: resX, type: "number", onChange: function (e) {
                                        setResX(Number(e.target.value));
                                    } }) })), _jsx(Grid, __assign({ item: true, xs: 3 }, { children: _jsx(TextField, { label: "resY", value: resY, type: "number", onChange: function (e) {
                                        setResY(Number(e.target.value));
                                    } }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(TextField, { InputProps: {
                                        startAdornment: (_jsx(InputAdornment, __assign({ position: "start" }, { children: "mm" }))),
                                    }, label: "pixelsize", value: pixelsize, type: "number", onChange: function (e) {
                                        setPixelsize(Number(e.target.value));
                                    } }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(TextField, { InputProps: {
                                        startAdornment: (_jsx(InputAdornment, __assign({ position: "start" }, { children: "mm" }))),
                                    }, label: "sampleDistance", value: sampleDistance, type: "number", onChange: function (e) {
                                        setSampleDistance(Number(e.target.value));
                                    } }) }))] }))] }), _jsx(CardActions, {})] })));
};
export default Instrumentation;
