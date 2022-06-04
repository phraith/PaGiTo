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
import Box from '@mui/material/Box/Box';
import { useEffect, useRef } from 'react';
import { Coordinate, LineProfileState, RelativeLineProfile } from '../../lib/LineProfile';
var LineProfileWrapper = function (props) {
    var canvasRef = useRef(null);
    // const [currentLineprofile, setCurrentLineprofile] = useState<RelativeLineProfile>(new RelativeLineProfile(new Coordinate(0, 0), new Coordinate(0, 1), new Coordinate(0,0)) );
    // const [lineMode, setLineMode] = useState<boolean>(false);
    // const [lineprofiles, setLineprofiles] = useState<RelativeLineProfile[]>([])
    var getMousePos = function (canvas, evt) {
        var bounds = canvas.getBoundingClientRect();
        // get the mouse coordinates, subtract the canvas top left and any scrolling
        var x = evt.pageX - bounds.left - scrollX;
        var y = evt.pageY - bounds.top - scrollY;
        x /= bounds.width;
        y /= bounds.height;
        x *= canvas.width;
        y *= canvas.height;
        return { x: x, y: y };
    };
    var initializeCanvas = function () {
        var canvas = canvasRef.current;
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        return canvas;
    };
    var createLineProfile = function (width, height, x, y) {
        var dim = new Coordinate(width, height);
        if (props.profileState.lineMode) {
            var start_1 = new Coordinate(x, 0);
            var end_1 = new Coordinate(x, height);
            return new RelativeLineProfile(start_1, end_1, dim);
        }
        var start = new Coordinate(0, y);
        var end = new Coordinate(width, y);
        return new RelativeLineProfile(start, end, dim);
    };
    var draw = function () {
        var canvas = initializeCanvas();
        var ctx = canvas.getContext("2d");
        if (ctx !== null) {
            var ctxSafe_1 = ctx;
            props.profileState.lineProfiles.forEach(function (staticLp) {
                drawLine(staticLp.toLineProfile(canvas.width, canvas.height), ctxSafe_1);
            });
            drawLine(props.profileState.currentLineProfile.toLineProfile(canvas.width, canvas.height), ctxSafe_1);
        }
    };
    var drawLine = function (lp, ctx) {
        ctx.beginPath();
        ctx.moveTo(lp.start.x, lp.start.y);
        ctx.lineTo(lp.end.x, lp.end.y);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2.5;
        ctx.stroke();
    };
    var handleMouseMove = function (e) {
        var canvas = canvasRef.current;
        var pos = getMousePos(canvas, e);
        var lp = createLineProfile(canvas.offsetWidth, canvas.offsetHeight, pos.x, pos.y);
        props.setProfileState(new LineProfileState(props.profileState.lineMode, props.profileState.lineProfiles, lp));
    };
    useEffect(function () {
        draw();
    }, [props.profileState]);
    var handleMousePress = function (e) {
        var canvas = canvasRef.current;
        var pos = getMousePos(canvas, e);
        var lp = createLineProfile(canvas.offsetWidth, canvas.offsetHeight, pos.x, pos.y);
        props.setProfileState(new LineProfileState(props.profileState.lineMode, __spreadArray(__spreadArray([], props.profileState.lineProfiles, true), [lp], false), props.profileState.currentLineProfile));
    };
    var handleKeyDown = function (event) {
        if (event.code === "KeyE") {
            props.setProfileState(new LineProfileState(!props.profileState.lineMode, props.profileState.lineProfiles, props.profileState.currentLineProfile));
        }
    };
    return (_jsx(Box, __assign({ onKeyDown: handleKeyDown }, { children: _jsxs(Box, __assign({ sx: { height: props.height, width: "100%", position: 'relative', zIndex: 0 } }, { children: [_jsx(Box, __assign({ sx: { position: 'relative', zIndex: 0 } }, { children: props.children })), _jsx(Box, __assign({ sx: { height: props.height, width: "100%", top: 0, left: 0, position: 'absolute', zIndex: 10 } }, { children: _jsx("canvas", { tabIndex: 1, onMouseDown: handleMousePress, onMouseMove: handleMouseMove, style: { height: props.height, width: "100%", position: "absolute" }, id: "canvas", ref: canvasRef }) }))] })) })));
};
export default LineProfileWrapper;
