let idsToDelete = [];
let rowsToUpdate = [];

function deleteRow(event, tableName, id) {
    event.preventDefault();
    idsToDelete.push(id);
    event.target.parentNode.parentNode.parentNode.remove();
}

function updateDate(event, tableName, id) {
    let newDate = event.target.value;
    rowsToUpdate.push({
        table_name: tableName,
        id: id,
        date: newDate
    });
}

function submitChanges(event) {
    event.preventDefault();
    fetch('/submit_changes', {
        method: 'POST',
        body: JSON.stringify({
            ids_to_delete: idsToDelete,
            rows_to_update: rowsToUpdate
        }),
        headers: {
            'Content-Type': 'application/json'
        }
    }).then(response => response.json()).then(data => {
        if (data.status === 'success') {
            alert('Changes saved successfully!');
        } else {
            alert('Failed to save changes: ' + data.message);
        }

        idsToDelete = [];
        rowsToUpdate = [];
    });
}
