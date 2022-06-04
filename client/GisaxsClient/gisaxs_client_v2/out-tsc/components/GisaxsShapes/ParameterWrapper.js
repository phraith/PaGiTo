import { jsx as _jsx } from "react/jsx-runtime";
import TextField from '@mui/material/TextField';
var ParameterWrapper = function (props) {
    return (_jsx(TextField, { label: props.parameterName, type: "number", onChange: function (e) {
            props.valueSetter(Number(e.target.value));
        }, variant: "outlined", defaultValue: props.defaultValue }));
};
export default ParameterWrapper;
