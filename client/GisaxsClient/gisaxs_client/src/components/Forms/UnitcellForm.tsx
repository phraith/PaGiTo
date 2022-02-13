import JSONForm from "@rjsf/material-ui";
import type { JSONSchema7 } from "json-schema";
import type { UiSchema } from "@rjsf/core";
import CustomObjectFieldTemplate from "./CustomObjectFieldTemplate"

import unitcellMeta from "./unitcell_meta.schema.json"

const uiSchema: UiSchema = {
    unitcellMeta: {
        "ui:ObjectFieldTemplate": CustomObjectFieldTemplate,
        repetitions: {
            x: {
                "ui:widget": "range",
            },
            y: {
                "ui:widget": "range",
            },
            z: {
                "ui:widget": "range",
            },
        },
        translation: {
            x: {
                "ui:widget": "range",
            },
            y: {
                "ui:widget": "range",
            },
            z: {
                "ui:widget": "range",
            },
        }
    }
};

const schema: any = unitcellMeta


function UnitcellForm(props: { callback: Function, formData: any }) {
    const changeHandler = (value: any) => {
        props.callback(value.formData);
    };

    return (
            <JSONForm
                children={true}
                schema={schema}
                formData={props.formData}
                uiSchema={uiSchema}
                onChange={(value) => { changeHandler(value) }}
                />
    );
}

export default UnitcellForm;
