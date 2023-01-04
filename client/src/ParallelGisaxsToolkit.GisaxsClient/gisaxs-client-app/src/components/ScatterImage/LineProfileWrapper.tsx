import Box from '@mui/material/Box/Box';
import { useEffect, useRef } from 'react';
import { LineProfile, Coordinate, LineProfileState, LineMode } from '../../utility/LineProfile';

interface LineprofileWrapperProps {
    width: any;
    height: any;
    profileState: LineProfileState;
    setProfileState: any;
    children: React.ReactNode;
}

const LineProfileWrapper : React.FC<LineprofileWrapperProps> = (props: LineprofileWrapperProps ) => {
    const canvasRef: any = useRef(null)

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

    const createLineProfile = (canvasWidth: number, canvasHeight: number, x: number, y: number) => {
        if (props.profileState.lineMode === LineMode.Vertical) {
            let start: Coordinate = new Coordinate((x / canvasWidth) * props.width, 0.0);
            let end: Coordinate = new Coordinate((x / canvasWidth) * props.width, props.height);
            return new LineProfile(start, end);
        }

        let start: Coordinate = new Coordinate(0.0, (y / canvasHeight) * props.height);
        let end: Coordinate = new Coordinate(props.width, (y / canvasHeight) * props.height);
        return new LineProfile(start, end);
    }

    const draw = () => {
        let canvas = initializeCanvas();
        let ctx: CanvasRenderingContext2D | null = canvas.getContext("2d");

        if (ctx !== null)
        {
            let ctxSafe: CanvasRenderingContext2D = ctx;
            props.profileState.lineProfiles.forEach(staticLp => {
                let canvasStart = new Coordinate((staticLp.start.x / props.width) * canvas.width, (staticLp.start.y / props.height) * canvas.height)
                let canvasEnd = new Coordinate((staticLp.end.x / props.width) * canvas.width, (staticLp.end.y / props.height) * canvas.height)
                drawLine(new LineProfile(canvasStart, canvasEnd), ctxSafe);
            })

            let canvasStart = new Coordinate((props.profileState.currentLineProfile.start.x / props.width) * canvas.width, (props.profileState.currentLineProfile.start.y / props.height) * canvas.height)
            let canvasEnd = new Coordinate((props.profileState.currentLineProfile.end.x / props.width) * canvas.width, (props.profileState.currentLineProfile.end.y / props.height) * canvas.height)
            drawLine(new LineProfile(canvasStart, canvasEnd), ctxSafe);
        }
    }

    const drawLine = (lp: LineProfile, ctx: CanvasRenderingContext2D) => {
        ctx.beginPath();
        ctx.moveTo(lp.start.x, lp.start.y);
        ctx.lineTo(lp.end.x, lp.end.y);
        ctx.strokeStyle = '#10a464';
        ctx.lineWidth = 2.5;
        ctx.stroke();
    }

    const handleMouseMove = (e) => {
        let canvas: HTMLCanvasElement = canvasRef.current
        let pos = getMousePos(canvas, e)
        let lp: LineProfile = createLineProfile(canvas.offsetWidth, canvas.offsetHeight, pos.x, pos.y)
        props.setProfileState(new LineProfileState(props.profileState.lineMode, props.profileState.lineProfiles, lp))
    }

    useEffect(() => {
        draw()
    }, [props.profileState]);

    const handleMousePress = (e) => {
        
        let canvas: HTMLCanvasElement = canvasRef.current
        let pos = getMousePos(canvas, e)
        let lp: LineProfile = createLineProfile(canvas.offsetWidth, canvas.offsetHeight, pos.x, pos.y)
        props.setProfileState(new LineProfileState(props.profileState.lineMode, [...props.profileState.lineProfiles, lp], props.profileState.currentLineProfile))
    }

    const handleKeyDown = (event: any): void => {
        if (event.code === "KeyE") {
            let newLineMode = props.profileState.lineMode === LineMode.Horizontal ? LineMode.Vertical : LineMode.Horizontal; 
            props.setProfileState(new LineProfileState(newLineMode, props.profileState.lineProfiles, props.profileState.currentLineProfile))
        }
    };
    return (
        <Box onKeyDown={handleKeyDown}>
          <Box sx={{ height: "100%", width: "100%", position: 'relative', zIndex: 0 }}>
            <Box sx={{ height: "100%", width: "100%", position: 'relative', zIndex: 0 }}>
              {props.children}
            </Box>
            <Box sx={{ height: "100%", width: "100%", top: 0, left: 0, position: 'absolute', zIndex: 10 }}>
              <canvas tabIndex={1} onMouseDown={handleMousePress} onMouseMove={handleMouseMove} style={{ height: "100%", width: "100%",  position: "absolute" }} id="canvas" ref={canvasRef}
              />
            </Box>
          </Box>
        </Box>
        );
}

export default LineProfileWrapper