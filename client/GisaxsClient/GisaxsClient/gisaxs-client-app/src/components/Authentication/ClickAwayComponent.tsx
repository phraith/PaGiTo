import ClickAwayListener from "@mui/base/ClickAwayListener/ClickAwayListener";
import Box from "@mui/material/Box/Box";
import Button from "@mui/material/Button/Button";
import React from "react";

interface ClickAwayComponentProps {
    description: string;
    children: React.ReactNode;
}

const ClickAwayComponent: React.FC<ClickAwayComponentProps> = (props: ClickAwayComponentProps) => {
    const [openForm, setOpenForm] = React.useState(false);
    return (
        <Box>
            {!openForm ? (
                <Button onClick={() => setOpenForm(true)} color="inherit">
                    {props.description}
                </Button>
            ) : (
                <ClickAwayListener onClickAway={() => setOpenForm(false)}>
                    <Box sx={{ position: "fixed" }}>
                        {props.children}
                    </Box>
                </ClickAwayListener>
            )}
        </Box>)
}

export default ClickAwayComponent