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
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import React, { useEffect } from "react";
import { SetLocalStorageEntity, UnitcellMetaConfig, } from "../Utility/DefaultConfigs";
var UnitcellMeta = function (props) {
    var _a = React.useState(UnitcellMetaConfig.repetitions.x), repX = _a[0], setRepX = _a[1];
    var _b = React.useState(UnitcellMetaConfig.repetitions.y), repY = _b[0], setRepY = _b[1];
    var _c = React.useState(UnitcellMetaConfig.repetitions.z), repZ = _c[0], setRepZ = _c[1];
    var _d = React.useState(UnitcellMetaConfig.translation.x), posX = _d[0], setPosX = _d[1];
    var _e = React.useState(UnitcellMetaConfig.translation.y), posY = _e[0], setPosY = _e[1];
    var _f = React.useState(UnitcellMetaConfig.translation.z), posZ = _f[0], setPosZ = _f[1];
    var localStorageEntityName = "unitcellMetaConfig";
    var configFieldName = "unitcellMeta";
    useEffect(function () {
        var currentConfig = {
            repetitions: {
                x: repX,
                y: repY,
                z: repZ,
            },
            translation: {
                x: posX,
                y: posY,
                z: posZ,
            },
        };
        SetLocalStorageEntity(currentConfig, UnitcellMetaConfig, localStorageEntityName);
        props.jsonCallback(currentConfig, configFieldName);
    }, [repX, repY, repZ, posX, posY, posZ]);
    useEffect(function () {
        var data = localStorage.getItem(localStorageEntityName);
        if (data !== null) {
            var unitcellMetaConfig = JSON.parse(data);
            setRepX(unitcellMetaConfig.repetitions.x);
            setRepY(unitcellMetaConfig.repetitions.y);
            setRepZ(unitcellMetaConfig.repetitions.z);
            setPosX(unitcellMetaConfig.translation.x);
            setPosY(unitcellMetaConfig.translation.y);
            setPosZ(unitcellMetaConfig.translation.z);
        }
    }, []);
    return (_jsxs(Card, __assign({ sx: {} }, { children: [_jsxs(CardContent, { children: [_jsx(Typography, { children: "UnitcellMeta" }), _jsxs(Grid, __assign({ container: true, sx: { paddingTop: 2 }, rowSpacing: 2 }, { children: [_jsx(Grid, __assign({ item: true, xs: 4 }, { children: _jsx(TextField, { label: "repX", value: repX, type: "number", onChange: function (e) {
                                        setRepX(Number(e.target.value));
                                    } }) })), _jsx(Grid, __assign({ item: true, xs: 4 }, { children: _jsx(TextField, { label: "repY", value: repY, type: "number", onChange: function (e) {
                                        setRepY(Number(e.target.value));
                                    } }) })), _jsx(Grid, __assign({ item: true, xs: 4 }, { children: _jsx(TextField, { label: "repZ", value: repZ, type: "number", onChange: function (e) {
                                        setRepZ(Number(e.target.value));
                                    } }) })), _jsx(Grid, __assign({ item: true, xs: 4 }, { children: _jsx(TextField, { label: "posX", value: posX, type: "number", onChange: function (e) {
                                        setPosX(Number(e.target.value));
                                    } }) })), _jsx(Grid, __assign({ item: true, xs: 4 }, { children: _jsx(TextField, { label: "posY", value: posY, type: "number", onChange: function (e) {
                                        setPosY(Number(e.target.value));
                                    } }) })), _jsx(Grid, __assign({ item: true, xs: 4 }, { children: _jsx(TextField, { label: "posZ", value: posZ, type: "number", onChange: function (e) {
                                        setPosZ(Number(e.target.value));
                                    } }) }))] }))] }), _jsx(CardActions, {})] })));
};
export default UnitcellMeta;
