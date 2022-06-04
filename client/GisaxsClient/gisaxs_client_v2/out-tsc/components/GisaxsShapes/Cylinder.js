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
var Cylinder = function (props) {
    var _a = useState(props.initialConfig.radius.mean), rMean = _a[0], setRMean = _a[1];
    var _b = useState(props.initialConfig.radius.stddev), rStddev = _b[0], setRStddev = _b[1];
    var _c = useState(props.initialConfig.height.mean), hMean = _c[0], setHMean = _c[1];
    var _d = useState(props.initialConfig.height.stddev), hStddev = _d[0], setHStddev = _d[1];
    //fix locations
    var _e = useState(props.initialConfig.locations[0].x), posX = _e[0], setPosX = _e[1];
    var _f = useState(props.initialConfig.locations[0].y), posY = _f[0], setPosY = _f[1];
    var _g = useState(props.initialConfig.locations[0].z), posZ = _g[0], setPosZ = _g[1];
    var _h = useState(props.initialConfig.refraction.beta), refBeta = _h[0], setRefBeta = _h[1];
    var _j = useState(props.initialConfig.refraction.delta), refDelta = _j[0], setRefDelta = _j[1];
    var _k = useState(true), collapsed = _k[0], setCollapsed = _k[1];
    var handleButtonClick = function () {
        setCollapsed(!collapsed);
    };
    var handleRemove = function (event) {
        props.removeCallback();
    };
    useEffect(function () {
        props.jsonCallback({
            type: "cylinder",
            radius: {
                mean: rMean,
                stddev: rStddev,
            },
            height: {
                mean: hMean,
                stddev: hStddev,
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
    }, [rStddev, rMean, hStddev, hMean, posX, posY, posZ, refBeta, refDelta]);
    return (_jsxs(Card, __assign({ sx: {} }, { children: [_jsxs(CardContent, { children: [_jsxs(Grid, __assign({ container: true, sx: {
                            paddingBottom: collapsed ? 0 : 2,
                        } }, { children: [_jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsxs(Typography, __assign({ sx: { fontSize: 14 }, color: "text.secondary", gutterBottom: true }, { children: ["Cylinder ", collapsed ? "[".concat(rMean, ", ").concat(rStddev, "]") : ""] })) })), _jsx(Grid, __assign({ item: true, xs: 3 }, { children: _jsx(Button, __assign({ size: "small", onClick: handleButtonClick }, { children: collapsed ? _jsx(ExpandMore, {}) : _jsx(ExpandLess, {}) })) })), _jsx(Grid, __assign({ item: true, xs: 3 }, { children: _jsx(Button, __assign({ size: "small", onClick: handleRemove }, { children: _jsx(DeleteForever, {}) })) }))] })), _jsx(Collapse, __assign({ in: !collapsed }, { children: _jsx(FormControl, { children: _jsxs(Grid, __assign({ container: true, direction: "row", rowSpacing: 1 }, { children: [_jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: rMean, valueSetter: setRMean, parameterName: "radiusMean" }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: rStddev, valueSetter: setRStddev, parameterName: "radiusStddev" }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: hMean, valueSetter: setHMean, parameterName: "heightMean" }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: hStddev, valueSetter: setHStddev, parameterName: "heightStddev" }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: refDelta, valueSetter: setRefDelta, parameterName: "refDelta" }) })), _jsx(Grid, __assign({ item: true, xs: 6 }, { children: _jsx(ParameterWrapper, { defaultValue: refBeta, valueSetter: setRefBeta, parameterName: "refBeta" }) })), _jsx(Grid, __assign({ item: true, xs: 4 }, { children: _jsx(ParameterWrapper, { defaultValue: posX, valueSetter: setPosX, parameterName: "posX" }) })), _jsx(Grid, __assign({ item: true, xs: 4 }, { children: _jsx(ParameterWrapper, { defaultValue: posY, valueSetter: setPosY, parameterName: "posY" }) })), _jsx(Grid, __assign({ item: true, xs: 4 }, { children: _jsx(ParameterWrapper, { defaultValue: posZ, valueSetter: setPosZ, parameterName: "posZ" }) }))] })) }) }))] }), _jsx(CardActions, {})] }), props.id));
};
export default Cylinder;
