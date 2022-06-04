import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import Simulation from "./components/Simulation/Simulation";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Fitting from "./components/Fitting/Fitting";
import React from "react";
var App = function () {
    return (_jsx(React.Fragment, { children: _jsx(BrowserRouter, { children: _jsxs(Routes, { children: [_jsx(Route, { path: "/", element: _jsx(Simulation, {}) }), _jsx(Route, { path: "simulation", element: _jsx(Simulation, {}) }), _jsx(Route, { path: "fitting", element: _jsx(Fitting, {}) })] }) }) }));
};
export default App;
