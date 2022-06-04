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
import Box from "@mui/material/Box";
import CssBaseline from "@mui/material/CssBaseline";
import Grid from "@mui/material/Grid";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import { throttle } from "lodash";
import ScatterImage from "../ScatterImage/ScatterImage";
import GisaxsShapes from "../GisaxsShapes/GisaxsShapes";
import Instrumentation from "../Instrumentation/Instrumentation";
import UnitcellMeta from "../UnitcellMeta/UnitcellMeta";
import React, { useEffect, useRef, useState } from "react";
import Sample from "../Sample/Sample";
import { HttpTransportType, HubConnectionBuilder, HubConnectionState, LogLevel, } from "@microsoft/signalr";
import LineProfileWrapper from "../ScatterImage/LineProfileWrapper";
import { Coordinate, LineProfileState, RelativeLineProfile } from "../../lib/LineProfile";
import LineProfileGraph from "./LineProfileGraph";
var Fitting = function () {
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
    var _a = useState(new HubConnectionBuilder()
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
    var _b = useState(), intensities = _b[0], setIntensities = _b[1];
    var _c = useState(), currentInfoPath = _c[0], setCurrentInfoPath = _c[1];
    var _d = useState(), imgWidth = _d[0], setImgWidth = _d[1];
    var _e = useState(), imgHeight = _e[0], setImgHeight = _e[1];
    var _f = useState(new LineProfileState(false, [], new RelativeLineProfile(new Coordinate(0, 0), new Coordinate(0, 1), new Coordinate(0, 0)))), lineprofileState = _f[0], setLineprofileState = _f[1];
    var _g = React.useState([]), plotData = _g[0], setPlotData = _g[1];
    useEffect(function () {
        if (connection) {
            connection
                .start()
                .then(function (result) {
                console.log("Connected!");
                connection.on("ReceiveJobId", function (message) {
                    receiveJobResult(message);
                });
                connection.on("ReceiveJobInfos", function (message) {
                    receiveJobInfos(message);
                });
                connection.on("ProcessLineprofiles", function (message) {
                    getLineprofiles(message);
                });
            })
                .catch(function (e) { return console.log("Connection failed: ", e); });
        }
    }, [connection]);
    var receiveJobInfos = function (message) {
        setCurrentInfoPath(message);
        setIsActive(true);
    };
    var getLineprofiles = function (message) {
        var j = JSON.parse(message);
        var traces = [];
        j["profiles"].forEach(function (element, i) {
            var values = element.Data;
            var k = values.map(function (x, index) { return { x: index, y: x }; });
            var trace = {
                data: k,
                id: "test",
                color: "hsl(155, 70%, 50%)"
            };
            traces.push(trace);
        });
        console.log("Traces", traces);
        setPlotData(traces);
    };
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
    var _h = React.useState("twilightShifted"), colormap = _h[0], setColorMap = _h[1];
    var _j = React.useState({}), jsonData = _j[0], setJsonData = _j[1];
    var _k = React.useState(false), isActive = _k[0], setIsActive = _k[1];
    var handleColorChange = function (event) {
        setColorMap(event.target.value);
    };
    var jsonCallback = function (value, key) {
        jsonData[key] = value;
        setJsonData(__assign({}, jsonData));
    };
    var sendLineprofileRequest = function (data, lineprofiles) {
        console.log("test", lineprofiles);
        var jsonConfig = JSON.stringify({
            profiles: __assign({}, lineprofiles),
            config: __assign({}, data),
        });
        if ((connection === null || connection === void 0 ? void 0 : connection.state) === HubConnectionState.Connected) {
            connection === null || connection === void 0 ? void 0 : connection.send("GetProfiles", jsonConfig);
            console.log("after profiles sent");
        }
    };
    var throttled = useRef(throttle(function (data, lineprofiles) { return sendLineprofileRequest(data, lineprofiles); }, 300));
    useEffect(function () {
        throttled.current(jsonData, [lineprofileState === null || lineprofileState === void 0 ? void 0 : lineprofileState.currentLineProfile]);
    }, [lineprofileState === null || lineprofileState === void 0 ? void 0 : lineprofileState.currentLineProfile]);
    // useEffect(() => {
    //   let interval: NodeJS.Timer | null = null;
    //   if (isActive) {
    //     interval = setInterval(() => {
    //       let url = "/api/redis/info?" + currentInfoPath;
    //       fetch(url, {
    //         method: "GET",
    //         headers: {
    //           Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
    //           Accept: "application/json",
    //         },
    //       })
    //         .then((response) => response.json())
    //         .then((data) => {
    //           let j = JSON.parse(data)
    //           let values = j.Values
    //           let k = values.map((x: number, index: number) => { return { x: index, y: x } })
    //           let l: { data: { x: number, y: number }[], id: string, color: string }[] = [{
    //             data: k,
    //             id: "test",
    //             color: "hsl(155, 70%, 50%)"
    //           }]
    //           setPlotData(l)
    //         }
    //         );
    //     }, 50);
    //   }
    // }, [isActive]);
    useEffect(function () {
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
    return (_jsxs(React.Fragment, { children: [_jsx(CssBaseline, {}), _jsx(MiniDrawer, {}), _jsxs(Grid, __assign({ container: true, spacing: 2 }, { children: [_jsx(Grid, __assign({ item: true, xs: 6, sm: 6, md: 6, lg: 4 }, { children: _jsx(Box, __assign({ sx: {
                                paddingTop: 10,
                                paddingLeft: 10,
                                paddingBottom: 10,
                            } }, { children: _jsx(LineProfileWrapper, __assign({ width: imgWidth, height: imgHeight, profileState: lineprofileState, setProfileState: function (state) {
                                    setLineprofileState(state);
                                } }, { children: _jsx(ScatterImage, { intensities: intensities, width: imgWidth, height: imgHeight }, "test") }), "test") })) })), _jsx(Grid, __assign({ item: true, xs: 6, sm: 6, md: 6, lg: 4 }, { children: _jsx(Box, { sx: {
                                paddingTop: 10,
                                paddingRight: 5,
                                paddingBottom: 10,
                            } }) })), _jsx(Grid, __assign({ item: true, xs: 12, sm: 12, md: 12, lg: 4 }, { children: _jsx(Grid, __assign({ container: true, sx: {
                                position: "sticky",
                                top: 0,
                                paddingTop: 10,
                                paddingRight: 5,
                                paddingLeft: 10,
                            } }, { children: _jsx(Grid, __assign({ item: true, xs: 12, sm: 12, md: 12, lg: 12 }, { children: _jsxs(Grid, __assign({ container: true, spacing: 2 }, { children: [_jsx(Grid, __assign({ item: true, xs: 12, sm: 7, md: 7, lg: 7 }, { children: _jsx(Instrumentation, { jsonCallback: jsonCallback }) })), _jsx(Grid, __assign({ item: true, xs: 12, sm: 5, md: 5, lg: 5 }, { children: _jsxs(Grid, __assign({ container: true, rowSpacing: 2 }, { children: [_jsx(Grid, __assign({ item: true, xs: 12 }, { children: _jsx(UnitcellMeta, { jsonCallback: jsonCallback }) })), _jsx(Grid, __assign({ item: true, xs: 12 }, { children: _jsx(Select, __assign({ value: colormap, onChange: handleColorChange }, { children: colors.map(function (value) { return (_jsx(MenuItem, __assign({ value: value }, { children: value }), value)); }) })) }))] })) })), _jsx(Grid, __assign({ item: true, xs: 12, sm: 7, md: 7, lg: 7 }, { children: _jsx(GisaxsShapes, { jsonCallback: jsonCallback }) })), _jsx(Grid, __assign({ item: true, xs: 12, sm: 5, md: 5, lg: 5 }, { children: _jsx(Sample, { jsonCallback: jsonCallback }) }))] })) })) })) }))] })), _jsx(LineProfileGraph, { plotData: plotData })] }));
};
export default Fitting;
