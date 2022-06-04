import Box from '@mui/material/Box/Box';
import { useEffect, useMemo, useRef, useState } from 'react';
import { LineProfile, Coordinate, LineProfileState, RelativeLineProfile } from '../../lib/LineProfile';
import debounce from 'lodash.debounce';

interface LineprofileWrapperProps {
    width: any;
    height: any;
    profileState: LineProfileState;
    setProfileState: any;
    children: React.ReactNode;
}

const LineProfileWrapper : React.FC<LineprofileWrapperProps> = (props: LineprofileWrapperProps ) => {
    const canvasRef: any = useRef(null)
    // const [currentLineprofile, setCurrentLineprofile] = useState<RelativeLineProfile>(new RelativeLineProfile(new Coordinate(0, 0), new Coordinate(0, 1), new Coordinate(0,0)) );
    // const [lineMode, setLineMode] = useState<boolean>(false);
    // const [lineprofiles, setLineprofiles] = useState<RelativeLineProfile[]>([])

    const getMousePos = (canvas, evt) => {
        var bounds = canvas.getBoundingClientRect();
        // get the mouse coordinates, subtract the canvas top left and any scrolling
        let x = evt.pageX - bounds.left - scrollX;
        let y = evt.pageY - bounds.top - scrollY;

        x /= bounds.width;
        y /= bounds.height;

        x *= canvas.width;
        y *= canvas.height;

        return { x: x, y: y }
    }


    const initializeCanvas = () => {
        let canvas: HTMLCanvasElement = canvasRef.current
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        return canvas;
    }

    const createLineProfile = (width: number, height: number, x: number, y: number) => {
        let dim: Coordinate = new Coordinate(width, height);
        if (props.profileState.lineMode) {
            let start: Coordinate = new Coordinate(x, 0);
            let end: Coordinate = new Coordinate(x, height);
            return new RelativeLineProfile(start, end, dim);
        }

        let start: Coordinate = new Coordinate(0, y);
        let end: Coordinate = new Coordinate(width, y);
        return new RelativeLineProfile(start, end, dim);
    }

    const draw = () => {
        let canvas = initializeCanvas();
        let ctx: CanvasRenderingContext2D | null = canvas.getContext("2d");

        if (ctx !== null)
        {
            let ctxSafe: CanvasRenderingContext2D = ctx;
            props.profileState.lineProfiles.forEach(staticLp => {
                drawLine(staticLp.toLineProfile(canvas.width, canvas.height), ctxSafe);
            })
            drawLine(props.profileState.currentLineProfile.toLineProfile(canvas.width, canvas.height), ctxSafe);
        }
    }

    const drawLine = (lp: LineProfile, ctx: CanvasRenderingContext2D) => {
        ctx.beginPath();
        ctx.moveTo(lp.start.x, lp.start.y);
        ctx.lineTo(lp.end.x, lp.end.y);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2.5;
        ctx.stroke();
    }

    const handleMouseMove = (e) => {
        let canvas: HTMLCanvasElement = canvasRef.current
        let pos = getMousePos(canvas, e)
        let lp: RelativeLineProfile = createLineProfile(canvas.offsetWidth, canvas.offsetHeight, pos.x, pos.y)
        props.setProfileState(new LineProfileState(props.profileState.lineMode, props.profileState.lineProfiles, lp))
    }

    useEffect(() => {
        draw()
    }, [props.profileState]);

    const handleMousePress = (e) => {
        
        let canvas: HTMLCanvasElement = canvasRef.current
        let pos = getMousePos(canvas, e)
        let lp: RelativeLineProfile = createLineProfile(canvas.offsetWidth, canvas.offsetHeight, pos.x, pos.y)
        props.setProfileState(new LineProfileState(props.profileState.lineMode, [...props.profileState.lineProfiles, lp], props.profileState.currentLineProfile))
    }

    const handleKeyDown = (event: any): void => {
        if (event.code === "KeyE") {
            props.setProfileState(new LineProfileState(!props.profileState.lineMode, props.profileState.lineProfiles, props.profileState.currentLineProfile))
        }
    };
    return (
    <Box onKeyDown={handleKeyDown}>
      <Box sx={{ height: props.height, width: "100%", position: 'relative', zIndex: 0 }}>
        <Box sx={{ position: 'relative', zIndex: 0 }}>
          {props.children}
        </Box>
        <Box sx={{ height: props.height, width: "100%", top: 0, left: 0, position: 'absolute', zIndex: 10 }}>
          <canvas tabIndex={1} onMouseDown={handleMousePress} onMouseMove={handleMouseMove} style={{ height: props.height, width: "100%",  position: "absolute" }} id="canvas" ref={canvasRef}
          />
        </Box>
      </Box>
    </Box>
    );
}

export default LineProfileWrapper