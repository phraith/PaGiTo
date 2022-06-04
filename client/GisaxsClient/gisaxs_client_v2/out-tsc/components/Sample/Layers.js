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
import React, { useEffect } from "react";
import { LayerConfig, SetLocalStorageEntity } from "../Utility/DefaultConfigs";
import { v4 as uuidv4 } from "uuid";
import Layer from "./Layer";
import Button from "@mui/material/Button";
import Grid from "@mui/material/Grid";
import Add from "@mui/icons-material/Add";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
var Layers = function (props) {
    var _a = React.useState([]), layers = _a[0], setLayers = _a[1];
    var _b = React.useState({}), jsonData = _b[0], setJsonData = _b[1];
    var localStorageEntityName = "layersConfig";
    var configFieldName = "layers";
    useEffect(function () {
        var formattedLayers = Object.keys(jsonData).map(function (key) { return jsonData[key]; });
        props.jsonCallback(formattedLayers, configFieldName);
        console.log("Set layersConfig");
        console.log(formattedLayers);
        SetLocalStorageEntity(formattedLayers, [], localStorageEntityName);
    }, [jsonData]);
    useEffect(function () {
        var data = localStorage.getItem(localStorageEntityName);
        console.log(data);
        if (data !== null) {
            var layersConfig = JSON.parse(data);
            var cachedLayers = [];
            for (var _i = 0, layersConfig_1 = layersConfig; _i < layersConfig_1.length; _i++) {
                var layer = layersConfig_1[_i];
                console.log(layer);
                cachedLayers = __spreadArray(__spreadArray([], cachedLayers, true), [createLayer(layer)], false);
            }
            setLayers(cachedLayers);
        }
    }, []);
    var removeLayer = function (id) {
        setLayers(function (layers) { return layers.filter(function (layer) { return layer.props.id !== id; }); });
        setJsonData(function (jsonData) {
            delete jsonData[id];
            return __assign({}, jsonData);
        });
    };
    var createJsonForLayer = function (layerJson, layerId) {
        setJsonData(function (jsonData) {
            jsonData[layerId] = layerJson;
            return __assign({}, jsonData);
        });
    };
    var createLayer = function (layerConfig) {
        var myid = uuidv4();
        return (_jsx(Layer, { id: myid, order: layerConfig.order == -1 ? layers.length : layerConfig.order, removeCallback: function () { return removeLayer(myid); }, initialConfig: layerConfig, jsonCallback: createJsonForLayer }, myid));
    };
    var addLayer = function () {
        setLayers(__spreadArray(__spreadArray([], layers, true), [createLayer(LayerConfig)], false));
    };
    return (_jsxs(Grid, __assign({ container: true }, { children: [_jsx(Grid, __assign({ item: true, xs: 12 }, { children: _jsx(List, { children: layers.map(function (value) {
                        return _jsx(ListItem, { children: value }, value.props.id);
                    }) }) })), _jsx(Grid, __assign({ item: true, xs: 4, sx: {
                    paddingBottom: layers.length === 0 ? 2 : 0,
                } }, { children: _jsx(Button, __assign({ size: "small", onClick: addLayer }, { children: _jsx(Add, {}) })) }))] })));
};
export default Layers;
