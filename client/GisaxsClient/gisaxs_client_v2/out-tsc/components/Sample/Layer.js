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
import DeleteForever from "@mui/icons-material/DeleteForever";
import ExpandLess from "@mui/icons-material/ExpandLess";
import ExpandMore from "@mui/icons-material/ExpandMore";
import Collapse from "@mui/material/Collapse";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActions from "@mui/material/CardActions";
import CardContent from "@mui/material/CardContent";
import FormControl from "@mui/material/FormControl";
import TextField from "@mui/material/TextField";
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";
import { useEffect, useState } from "react";
import ParameterWrapper from "../GisaxsShapes/ParameterWrapper";
var Layer = function (props) {
    var _a = useState(true), collapsed = _a[0], setCollapsed = _a[1];
    var _b = useState(props.initialConfig.thickness), thickness = _b[0], setThickness = _b[1];
    var _c = useState(props.initialConfig.refraction.beta), refBeta = _c[0], setRefBeta = _c[1];
    var _d = useState(props.initialConfig.refraction.delta), refDelta = _d[0], setRefDelta = _d[1];
    var handleButtonClick = function () {
        setCollapsed(!collapsed);
    };
    var handleRemove = function (event) {
        props.removeCallback();
    };
    useEffect(function () {
        props.jsonCallback({
            refraction: {
                delta: refDelta,
                beta: refBeta,
            },
            order: props.order,
            thickness: thickness
        }, props.id);
    }, [thickness, refBeta, refDelta]);
    return (_jsxs(Card, __assign({ sx: {} }, { children: [_jsxs(CardContent, { children: [_jsxs(Grid, __assign({ container: true, sx: {
                            paddingBottom: collapsed ? 0 : 2,
                        } }, { children: [_jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsxs(Typography, __assign({ sx: { fontSize: 14 }, color: "text.secondary", gutterBottom: true }, { children: ["Layer", " ", collapsed
                                            ? "[".concat(props.order, ", ").concat(thickness.toExponential(), ", ").concat(refBeta.toExponential(), ", ").concat(refDelta.toExponential(), "]")
                                            : ""] })) })), _jsx(Grid, __assign({ item: true, xs: 3 }, { children: _jsx(Button, __assign({ size: "small", onClick: handleButtonClick }, { children: collapsed ? _jsx(ExpandMore, {}) : _jsx(ExpandLess, {}) })) })), _jsx(Grid, __assign({ item: true, xs: 3 }, { children: _jsx(Button, __assign({ size: "small", onClick: handleRemove }, { children: _jsx(DeleteForever, {}) })) }))] })), _jsx(Collapse, __assign({ in: !collapsed }, { children: _jsx(FormControl, { children: _jsxs(Grid, __assign({ container: true, direction: "row", rowSpacing: 1 }, { children: [_jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(TextField, { label: "order", type: "number", variant: "outlined", inputProps: {
                                                readOnly: true,
                                                disabled: true
                                            }, value: props.order }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: thickness, valueSetter: setThickness, parameterName: "thickness" }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: refBeta, valueSetter: setRefBeta, parameterName: "beta" }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: refDelta, valueSetter: setRefDelta, parameterName: "delta" }) }))] })) }) }))] }), _jsx(CardActions, {})] }), props.id));
};
export default Layer;
