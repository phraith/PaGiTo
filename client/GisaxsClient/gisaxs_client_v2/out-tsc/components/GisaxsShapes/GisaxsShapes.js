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
var __spreadArray = (this && this.__spreadArray) || function (to, from, pack) {
    if (pack || arguments.length === 2) for (var i = 0, l = from.length, ar; i < l; i++) {
        if (ar || !(i in from)) {
            if (!ar) ar = Array.prototype.slice.call(from, 0, i);
            ar[i] = from[i];
        }
    }
    return to.concat(ar || Array.prototype.slice.call(from));
};
import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import Add from "@mui/icons-material/Add";
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import CardContent from "@mui/material/CardContent";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import Grid from "@mui/material/Grid";
import Typography from "@mui/material/Typography";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import React, { useEffect } from "react";
import Cylinder from "./Cylinder";
import Sphere from "./Sphere";
import { v4 as uuidv4 } from "uuid";
import { CylinderConfig, SetLocalStorageEntity, SphereConfig, } from "../Utility/DefaultConfigs";
var GisaxsShapes = function (props) {
    var _a = React.useState([]), shapes = _a[0], setShapes = _a[1];
    var _b = React.useState(null), anchorEl = _b[0], setAnchor = _b[1];
    var _c = React.useState({}), jsonData = _c[0], setJsonData = _c[1];
    var localStorageEntityName = "shapesConfig";
    var configFieldName = "shapes";
    useEffect(function () {
        var formattedShapes = Object.keys(jsonData).map(function (key) { return jsonData[key]; });
        props.jsonCallback(formattedShapes, configFieldName);
        SetLocalStorageEntity(formattedShapes, [], localStorageEntityName);
    }, [jsonData]);
    useEffect(function () {
        var data = localStorage.getItem(localStorageEntityName);
        if (data !== null) {
            var shapesConfig = JSON.parse(data);
            var cachedShapes = [];
            for (var _i = 0, shapesConfig_1 = shapesConfig; _i < shapesConfig_1.length; _i++) {
                var shape = shapesConfig_1[_i];
                switch (shape.type) {
                    case "sphere":
                        cachedShapes = __spreadArray(__spreadArray([], cachedShapes, true), [createSphere(shape)], false);
                        break;
                    case "cylinder":
                        cachedShapes = __spreadArray(__spreadArray([], cachedShapes, true), [createCylinder(shape)], false);
                        break;
                }
            }
            setShapes(cachedShapes);
        }
    }, []);
    var removeShape = function (id) {
        setShapes(function (shapes) { return shapes.filter(function (shape) { return shape.props.id !== id; }); });
        setJsonData(function (jsonData) {
            delete jsonData[id];
            return __assign({}, jsonData);
        });
    };
    var createJsonForSphere = function (sphereJson, shapeId) {
        setJsonData(function (jsonData) {
            jsonData[shapeId] = sphereJson;
            return __assign({}, jsonData);
        });
    };
    var addShape = function (e) {
        setAnchor(e.currentTarget);
    };
    var addSphere = function () {
        setShapes(__spreadArray(__spreadArray([], shapes, true), [createSphere(SphereConfig)], false));
        setAnchor(null);
    };
    var createSphere = function (sphereConfig) {
        var myid = uuidv4();
        return (_jsx(Sphere, { id: myid, removeCallback: function () { return removeShape(myid); }, jsonCallback: createJsonForSphere, initialConfig: sphereConfig }, myid));
    };
    var addCylinder = function () {
        setShapes(__spreadArray(__spreadArray([], shapes, true), [createCylinder(CylinderConfig)], false));
        setAnchor(null);
    };
    var createCylinder = function (cylinderConfig) {
        var myid = uuidv4();
        return (_jsx(Cylinder, { id: myid, removeCallback: function () { return removeShape(myid); }, jsonCallback: createJsonForSphere, initialConfig: cylinderConfig }));
    };
    var handleClose = function () {
        setAnchor(null);
    };
    return (_jsx(Card, __assign({ style: { maxHeight: 700, overflow: "auto" } }, { children: _jsxs(CardContent, { children: [_jsxs(Grid, __assign({ container: true }, { children: [_jsx(Grid, __assign({ item: true, xs: 8 }, { children: _jsx(Typography, { children: "GisaxsShapesConfig" }) })), _jsxs(Grid, __assign({ item: true, xs: 4 }, { children: [_jsx(Button, __assign({ size: "small", onClick: addShape }, { children: _jsx(Add, {}) })), _jsxs(Menu, __assign({ anchorEl: anchorEl, keepMounted: true, open: Boolean(anchorEl), onClose: handleClose }, { children: [_jsx(MenuItem, __assign({ onClick: addSphere }, { children: "Sphere" })), _jsx(MenuItem, __assign({ onClick: addCylinder }, { children: "Cylinder" }))] }))] }))] })), _jsx(List, { children: shapes.map(function (value) {
                        return _jsx(ListItem, { children: value }, value.props.id);
                    }) })] }) })));
};
export default GisaxsShapes;
