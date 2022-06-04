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
import MiniDrawer from "../Drawer/MiniDrawer";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import Grid from "@mui/material/Grid";
import Select from "@mui/material/Select";
import Box from "@mui/material/Box";
import CssBaseline from "@mui/material/CssBaseline";
import ScatterImage from "../ScatterImage/ScatterImage";
import GisaxsShapes from "../GisaxsShapes/GisaxsShapes";
import Instrumentation from "../Instrumentation/Instrumentation";
import UnitcellMeta from "../UnitcellMeta/UnitcellMeta";
import React, * as react from "react";
import Sample from "../Sample/Sample";
import { HttpTransportType, HubConnectionBuilder, HubConnectionState, LogLevel, } from "@microsoft/signalr";
var Simulation = function () {
    var colors = [
        "twilightShifted",
        "twilight",
        "autumn",
        "parula",
        "bone",
        "cividis",
        "cool",
        "hot",
        "hsv",
        "inferno",
        "jet",
        "magma",
        "ocean",
        "pink",
        "plasma",
        "rainbow",
        "spring",
        "summer",
        "viridis",
        "winter",
    ];
    var _a = react.useState(new HubConnectionBuilder()
        .withUrl("/message", {
        skipNegotiation: true,
        transport: HttpTransportType.WebSockets,
        accessTokenFactory: function () {
            return "".concat(localStorage.getItem("apiToken"));
        },
    })
        .configureLogging(LogLevel.Information)
        .withAutomaticReconnect()
        .build()), connection = _a[0], _ = _a[1];
    var _b = react.useState(), intensities = _b[0], setIntensities = _b[1];
    var _c = react.useState(), imgWidth = _c[0], setImgWidth = _c[1];
    var _d = react.useState(), imgHeight = _d[0], setImgHeight = _d[1];
    react.useEffect(function () {
        if (connection) {
            connection
                .start()
                .then(function (result) {
                console.log("Connected!");
                connection.on("ReceiveJobId", function (message) {
                    receiveJobResult(message);
                });
            })
                .catch(function (e) { return console.log("Connection failed: ", e); });
        }
    }, [connection]);
    var receiveJobResult = function (message) {
        var url = "/api/redis/data?" + message;
        fetch(url, {
            method: "GET",
            headers: {
                Authorization: "Bearer ".concat(localStorage.getItem("apiToken")),
                Accept: "application/json",
            },
        })
            .then(function (response) { return response.json(); })
            .then(function (data) { return handleData(data); });
    };
    var handleData = function (input) {
        var startTime = performance.now();
        var json = JSON.parse(input);
        setIntensities(json.data);
        setImgWidth(json.width);
        setImgHeight(json.height);
        var endTime = performance.now();
        console.log("Handling data took ".concat(endTime - startTime, " milliseconds"));
    };
    var _e = React.useState("twilightShifted"), colormap = _e[0], setColorMap = _e[1];
    var _f = React.useState({}), jsonData = _f[0], setJsonData = _f[1];
    var handleColorChange = function (event) {
        setColorMap(event.target.value);
    };
    var jsonCallback = function (value, key) {
        jsonData[key] = value;
        setJsonData(__assign({}, jsonData));
    };
    react.useEffect(function () {
        var jsonConfig = JSON.stringify({
            info: {
                clientId: 0,
                jobId: 0,
                jobType: "sim",
                colormapName: colormap,
            },
            config: __assign({}, jsonData),
        });
        localStorage.setItem("simulation_config", jsonConfig);
        if ((connection === null || connection === void 0 ? void 0 : connection.state) === HubConnectionState.Connected) {
            connection === null || connection === void 0 ? void 0 : connection.send("IssueJob", jsonConfig);
            console.log("after job sent");
        }
    }, [jsonData, colormap]);
    return (_jsxs(React.Fragment, { children: [_jsx(CssBaseline, {}), _jsx(MiniDrawer, {}), _jsxs(Grid, __assign({ container: true, spacing: 2 }, { children: [_jsx(Grid, __assign({ item: true, xs: 12, sm: 12, md: 12, lg: 8 }, { children: _jsx(Box, __assign({ sx: {
                                paddingTop: 10,
                                paddingRight: 5,
                                paddingLeft: 10,
                                paddingBottom: 10,
                            } }, { children: _jsx(ScatterImage, { intensities: intensities, width: imgWidth, height: imgHeight }) })) })), _jsx(Grid, __assign({ item: true, xs: 12, sm: 12, md: 12, lg: 4 }, { children: _jsx(Grid, __assign({ container: true, sx: {
                                position: "sticky",
                                top: 0,
                                paddingTop: 10,
                                paddingRight: 5,
                                paddingLeft: 10,
                            } }, { children: _jsx(Grid, __assign({ item: true, xs: 12, sm: 12, md: 12, lg: 12 }, { children: _jsxs(Grid, __assign({ container: true, spacing: 2 }, { children: [_jsx(Grid, __assign({ item: true, xs: 12, sm: 7, md: 7, lg: 7 }, { children: _jsx(Instrumentation, { jsonCallback: jsonCallback }) })), _jsx(Grid, __assign({ item: true, xs: 12, sm: 5, md: 5, lg: 5 }, { children: _jsxs(Grid, __assign({ container: true, rowSpacing: 2 }, { children: [_jsx(Grid, __assign({ item: true, xs: 12 }, { children: _jsx(UnitcellMeta, { jsonCallback: jsonCallback }) })), _jsx(Grid, __assign({ item: true, xs: 12 }, { children: _jsx(FormControl, __assign({ fullWidth: true }, { children: _jsx(Select, __assign({ value: colormap, onChange: handleColorChange }, { children: colors.map(function (value) { return (_jsx(MenuItem, __assign({ value: value }, { children: value }), value)); }) })) })) }))] })) })), _jsx(Grid, __assign({ item: true, xs: 12, sm: 7, md: 7, lg: 7 }, { children: _jsx(GisaxsShapes, { jsonCallback: jsonCallback }) })), _jsx(Grid, __assign({ item: true, xs: 12, sm: 5, md: 5, lg: 5 }, { children: _jsx(Sample, { jsonCallback: jsonCallback }) }))] })) })) })) }))] }))] }));
};
export default Simulation;
