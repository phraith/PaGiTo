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
import { jsxs as _jsxs, jsx as _jsx } from "react/jsx-runtime";
import ExpandLess from "@mui/icons-material/ExpandLess";
import ExpandMore from "@mui/icons-material/ExpandMore";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActions from "@mui/material/CardActions";
import CardContent from "@mui/material/CardContent";
import Collapse from "@mui/material/Collapse";
import FormControl from "@mui/material/FormControl";
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import { useEffect, useState } from "react";
var Substrate = function (props) {
    var _a = useState(true), collapsed = _a[0], setCollapsed = _a[1];
    var _b = useState(2e-8), refBeta = _b[0], setRefBeta = _b[1];
    var _c = useState(6e-6), refDelta = _c[0], setRefDelta = _c[1];
    var handleButtonClick = function () {
        setCollapsed(!collapsed);
    };
    useEffect(function () {
        props.jsonCallback({
            refraction: {
                delta: refDelta,
                beta: refBeta,
            }
        }, "substrate");
    }, [refBeta, refDelta]);
    return (_jsxs(Card, __assign({ sx: {} }, { children: [_jsxs(CardContent, { children: [_jsxs(Grid, __assign({ container: true, sx: {
                            paddingBottom: collapsed ? 0 : 2,
                        } }, { children: [_jsx(Grid, __assign({ item: true, xs: 10 }, { children: _jsxs(Typography, __assign({ sx: { fontSize: 14 }, color: "text.secondary", gutterBottom: true }, { children: ["Substrate ", collapsed ? "[".concat(refBeta.toExponential(), ", ").concat(refDelta.toExponential(), "]") : ""] })) })), _jsx(Grid, __assign({ item: true, xs: 2 }, { children: _jsx(Button, __assign({ size: "small", onClick: handleButtonClick }, { children: collapsed ? _jsx(ExpandMore, {}) : _jsx(ExpandLess, {}) })) }))] })), _jsx(Collapse, __assign({ in: !collapsed }, { children: _jsx(FormControl, { children: _jsxs(Grid, __assign({ container: true, direction: "row", rowSpacing: 1 }, { children: [_jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(TextField, { type: "number", label: "refBeta", onChange: function (e) { return setRefBeta(Number(e.target.value)); }, variant: "outlined", defaultValue: refBeta }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(TextField, { type: "number", label: "refDelta", onChange: function (e) { return setRefDelta(Number(e.target.value)); }, variant: "outlined", defaultValue: refDelta }) }))] })) }) }))] }), _jsx(CardActions, {})] })));
};
export default Substrate;
