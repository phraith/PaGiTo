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
import CardContent from "@mui/material/CardContent";
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";
import React, { useEffect } from "react";
import Substrate from "./Substrate";
import Layers from "./Layers";
var Sample = function (props) {
    var _a = React.useState({
        layers: {}
    }), jsonData = _a[0], setJsonData = _a[1];
    var configFieldName = "sample";
    useEffect(function () {
        props.jsonCallback(jsonData, configFieldName);
    }, [jsonData]);
    var jsonCallback = function (value, key) {
        jsonData[key] = value;
        setJsonData(__assign({}, jsonData));
    };
    return (_jsx(Card, __assign({ style: { maxHeight: 700, overflow: "auto" } }, { children: _jsx(CardContent, { children: _jsxs(Grid, __assign({ container: true, rowSpacing: 2 }, { children: [_jsx(Grid, __assign({ item: true, xs: 8 }, { children: _jsx(Typography, { children: "Sample" }) })), _jsx(Grid, __assign({ item: true, xs: 12 }, { children: _jsx(Substrate, { jsonCallback: jsonCallback }) })), _jsx(Grid, __assign({ item: true, xs: 12 }, { children: _jsx(Layers, { jsonCallback: jsonCallback }) }))] })) }) })));
};
export default Sample;
