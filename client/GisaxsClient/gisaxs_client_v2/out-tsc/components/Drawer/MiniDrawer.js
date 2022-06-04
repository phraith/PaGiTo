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
import * as React from "react";
import { styled, useTheme } from "@mui/material/styles";
import Box from "@mui/material/Box";
import MuiDrawer from "@mui/material/Drawer";
import MuiAppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import List from "@mui/material/List";
import CssBaseline from "@mui/material/CssBaseline";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import ListItemIcon from "@mui/material/ListItemIcon";
import ListItemText from "@mui/material/ListItemText";
import DeviceHubIcon from "@mui/icons-material/DeviceHub";
import TimelineIcon from "@mui/icons-material/Timeline";
import Button from "@mui/material/Button";
import ClickAwayListener from "@mui/material/ClickAwayListener";
import Grid from "@mui/material/Grid";
import ListItem from "@mui/material/ListItem";
import Login from "../Login/Login";
import { Link } from "react-router-dom";
var drawerWidth = 240;
var openedMixin = function (theme) { return ({
    width: drawerWidth,
    transition: theme.transitions.create("width", {
        easing: theme.transitions.easing.sharp,
        duration: theme.transitions.duration.enteringScreen,
    }),
    overflowX: "hidden",
}); };
var closedMixin = function (theme) {
    var _a;
    return (_a = {
            transition: theme.transitions.create("width", {
                easing: theme.transitions.easing.sharp,
                duration: theme.transitions.duration.leavingScreen,
            }),
            overflowX: "hidden",
            width: "calc(".concat(theme.spacing(7), " + 1px)")
        },
        _a[theme.breakpoints.up("sm")] = {
            width: "calc(".concat(theme.spacing(8), " + 1px)"),
        },
        _a);
};
var DrawerHeader = styled("div")(function (_a) {
    var theme = _a.theme;
    return (__assign({ display: "flex", alignItems: "center", justifyContent: "flex-end", padding: theme.spacing(0, 1) }, theme.mixins.toolbar));
});
var AppBar = styled(MuiAppBar, {
    shouldForwardProp: function (prop) { return prop !== "open"; },
})(function (_a) {
    var theme = _a.theme, open = _a.open;
    return (__assign({ zIndex: theme.zIndex.drawer + 1, transition: theme.transitions.create(["width", "margin"], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.leavingScreen,
        }) }, (open && {
        marginLeft: drawerWidth,
        width: "calc(100% - ".concat(drawerWidth, "px)"),
        transition: theme.transitions.create(["width", "margin"], {
            easing: theme.transitions.easing.sharp,
            duration: theme.transitions.duration.enteringScreen,
        }),
    })));
});
var Drawer = styled(MuiDrawer, {
    shouldForwardProp: function (prop) { return prop !== "open"; },
})(function (_a) {
    var theme = _a.theme, open = _a.open;
    return (__assign(__assign({ width: drawerWidth, flexShrink: 0, whiteSpace: "nowrap", boxSizing: "border-box" }, (open && __assign(__assign({}, openedMixin(theme)), { "& .MuiDrawer-paper": openedMixin(theme) }))), (!open && __assign(__assign({}, closedMixin(theme)), { "& .MuiDrawer-paper": closedMixin(theme) }))));
});
export default function MiniDrawer() {
    var theme = useTheme();
    var _a = React.useState(false), open = _a[0], setOpen = _a[1];
    var _b = React.useState(false), openLoginForm = _b[0], setOpenLoginForm = _b[1];
    var handleDrawerOpen = function () {
        setOpen(true);
    };
    var handleDrawerClose = function () {
        setOpen(false);
    };
    var openLoginMenu = function () {
        setOpenLoginForm(true);
    };
    var closeLoginMenu = function () {
        setOpenLoginForm(false);
    };
    var FittingLink = React.forwardRef(function (props, ref) { return _jsx(Link, __assign({ ref: ref, to: "/fitting" }, props)); });
    var SimulationLink = React.forwardRef(function (props, ref) { return _jsx(Link, __assign({ ref: ref, to: "/simulation" }, props)); });
    return (_jsxs(Box, __assign({ sx: { display: "flex" } }, { children: [_jsx(CssBaseline, {}), _jsx(AppBar, __assign({ position: "fixed", open: open }, { children: _jsxs(Toolbar, { children: [_jsx(IconButton, __assign({ color: "inherit", "aria-label": "open drawer", onClick: handleDrawerOpen, edge: "start", sx: __assign({ marginRight: 5 }, (open && { display: "none" })) }, { children: _jsx(MenuIcon, {}) })), _jsxs(Grid, __assign({ container: true, justifyContent: "space-between" }, { children: [_jsx(Grid, { item: true }), _jsx(Grid, __assign({ item: true, xs: 3, md: 3, lg: 1 }, { children: !openLoginForm ? (_jsx(Button, __assign({ onClick: openLoginMenu, color: "inherit" }, { children: "Auth" }))) : (_jsx(ClickAwayListener, __assign({ onClickAway: closeLoginMenu }, { children: _jsx(Box, __assign({ sx: { position: "fixed" } }, { children: _jsx(Login, {}) })) }))) }))] }))] }) })), _jsxs(Drawer, __assign({ variant: "permanent", open: open }, { children: [_jsx(DrawerHeader, { children: _jsx(IconButton, __assign({ onClick: handleDrawerClose }, { children: theme.direction === "rtl" ? (_jsx(ChevronRightIcon, {})) : (_jsx(ChevronLeftIcon, {})) })) }), _jsx(Divider, {}), _jsxs(List, { children: [_jsxs(ListItem, __assign({ component: SimulationLink, sx: {
                                    minHeight: 48,
                                    justifyContent: open ? "initial" : "center",
                                    px: 2.5,
                                } }, { children: [_jsx(ListItemText, { primary: "ModelSimulation", sx: { opacity: open ? 1 : 0 } }), _jsx(ListItemIcon, __assign({ sx: {
                                            minWidth: 0,
                                            mr: open ? 3 : "auto",
                                            justifyContent: "center",
                                        } }, { children: _jsx(DeviceHubIcon, {}) }))] })), _jsxs(ListItem, __assign({ component: FittingLink, sx: {
                                    minHeight: 48,
                                    justifyContent: open ? "initial" : "center",
                                    px: 2.5,
                                } }, { children: [_jsx(ListItemText, { primary: "ModelFitting", sx: { opacity: open ? 1 : 0 } }), _jsx(ListItemIcon, __assign({ sx: {
                                            minWidth: 0,
                                            mr: open ? 3 : "auto",
                                            justifyContent: "center",
                                        } }, { children: _jsx(TimelineIcon, {}) }))] }), "ModelFitting")] })] }))] })));
}
