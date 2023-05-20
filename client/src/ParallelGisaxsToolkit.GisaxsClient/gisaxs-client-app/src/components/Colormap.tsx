import FormControl from "@mui/material/FormControl/FormControl";
import MenuItem from "@mui/material/MenuItem/MenuItem";
import Select from "@mui/material/Select/Select";

interface ColormapSelectProps {
    colormap: any;
    setColormap: any;
}

const ColormapSelect = (props: ColormapSelectProps) => {
    const colors = [
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

    return (
        <FormControl>
            <Select value={props.colormap} onChange={(event) => { props.setColormap(event.target.value as string) }}>
                {colors.map((value) => (
                    <MenuItem key={value} value={value}>
                        {value}
                    </MenuItem>
                ))}
            </Select>
        </FormControl>
    );
}

export default ColormapSelect;