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
import Button from "@mui/material/Button";
import Card from "@mui/material/Card";
import TextField from "@mui/material/TextField";
import Stack from "@mui/material/Stack";
import ListItem from "@mui/material/ListItem";
import React from "react";
var Login = function () {
    var _a = React.useState(""), currentPassword = _a[0], setCurrentPassword = _a[1];
    var _b = React.useState(""), currentUsername = _b[0], setCurrentUsername = _b[1];
    var handleLogin = function () {
        var requestOptions = {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                username: currentUsername,
                password: currentPassword,
            }),
        };
        var url = "/api/auth/login";
        fetch(url, requestOptions).then(function (response) { return response.text(); }).then(function (token) {
            localStorage.setItem('apiToken', token);
        });
    };
    return (_jsx(Card, { children: _jsxs(Stack, { children: [_jsx(ListItem, { children: _jsx(TextField, __assign({ id: "outlined-username-input", label: "Username", onChange: function (e) { return setCurrentUsername(e.target.value); } }, { children: "User" })) }), _jsx(ListItem, { children: _jsx(TextField, __assign({ type: "password", id: "outlined-password-input", label: "Password", onChange: function (e) { return setCurrentPassword(e.target.value); } }, { children: "Password" })) }), _jsx(ListItem, { children: _jsx(Button, __assign({ onClick: handleLogin }, { children: "Login" })) })] }) }));
};
export default Login;
