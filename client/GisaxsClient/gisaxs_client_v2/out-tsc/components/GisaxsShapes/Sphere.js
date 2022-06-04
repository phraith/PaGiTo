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
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardActions from "@mui/material/CardActions";
import CardContent from "@mui/material/CardContent";
import Collapse from "@mui/material/Collapse";
import FormControl from "@mui/material/FormControl";
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";
import { useEffect, useState } from "react";
import ParameterWrapper from "./ParameterWrapper";
var Sphere = function (props) {
    var _a = useState(true), collapsed = _a[0], setCollapsed = _a[1];
    var _b = useState(props.initialConfig.radius.mean), rMean = _b[0], setRMean = _b[1];
    var _c = useState(props.initialConfig.radius.stddev), rStddev = _c[0], setRStddev = _c[1];
    //fix locations
    var _d = useState(props.initialConfig.locations[0].x), posX = _d[0], setPosX = _d[1];
    var _e = useState(props.initialConfig.locations[0].y), posY = _e[0], setPosY = _e[1];
    var _f = useState(props.initialConfig.locations[0].z), posZ = _f[0], setPosZ = _f[1];
    var _g = useState(props.initialConfig.refraction.beta), refBeta = _g[0], setRefBeta = _g[1];
    var _h = useState(props.initialConfig.refraction.delta), refDelta = _h[0], setRefDelta = _h[1];
    var handleButtonClick = function () {
        setCollapsed(!collapsed);
    };
    var handleRemove = function (event) {
        props.removeCallback();
    };
    useEffect(function () {
        props.jsonCallback({
            type: "sphere",
            radius: {
                mean: rMean,
                stddev: rStddev,
            },
            refraction: {
                delta: refDelta,
                beta: refBeta,
            },
            locations: [
                {
                    x: posX,
                    y: posY,
                    z: posZ,
                },
            ],
        }, props.id);
    }, [rStddev, rMean, posX, posY, posZ, refBeta, refDelta]);
    return (_jsxs(Card, __assign({ sx: {} }, { children: [_jsxs(CardContent, { children: [_jsxs(Grid, __assign({ container: true, sx: {
                            paddingBottom: collapsed ? 0 : 2,
                        } }, { children: [_jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsxs(Typography, __assign({ sx: { fontSize: 14 }, color: "text.secondary", gutterBottom: true }, { children: ["Sphere ", collapsed ? "[".concat(rMean, ", ").concat(rStddev, "]") : ""] })) })), _jsx(Grid, __assign({ item: true, xs: 3 }, { children: _jsx(Button, __assign({ size: "small", onClick: handleButtonClick }, { children: collapsed ? _jsx(ExpandMore, {}) : _jsx(ExpandLess, {}) })) })), _jsx(Grid, __assign({ item: true, xs: 3 }, { children: _jsx(Button, __assign({ size: "small", onClick: handleRemove }, { children: _jsx(DeleteForever, {}) })) }))] })), _jsx(Collapse, __assign({ in: !collapsed }, { children: _jsx(FormControl, { children: _jsxs(Grid, __assign({ container: true, direction: "row", rowSpacing: 1 }, { children: [_jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: rMean, valueSetter: setRMean, parameterName: "radiusMean" }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: rStddev, valueSetter: setRStddev, parameterName: "radiusStddev" }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: refDelta, valueSetter: setRefDelta, parameterName: "refDelta" }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: refBeta, valueSetter: setRefBeta, parameterName: "refBeta" }) })), _jsx(Grid, __assign({ item: true, xs: 4 }, { children: _jsx(ParameterWrapper, { defaultValue: posX, valueSetter: setPosX, parameterName: "posX" }) })), _jsx(Grid, __assign({ item: true, xs: 4 }, { children: _jsx(ParameterWrapper, { defaultValue: posY, valueSetter: setPosY, parameterName: "posY" }) })), _jsx(Grid, __assign({ item: true, xs: 4 }, { children: _jsx(ParameterWrapper, { defaultValue: posZ, valueSetter: setPosZ, parameterName: "posZ" }) }))] })) }) }))] }), _jsx(CardActions, {})] }), props.id));
};
export default Sphere;
